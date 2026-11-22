/*
 * GazeAssistGlasses.kt: integration skeleton for Android XR audio glasses.
 *
 * STATUS: untested. Not compiled or run on the emulator or on real glasses.
 * The risk engine below is a direct port of gaze_assist.py thresholds and
 * wording. The glasses-camera binding follows the Jetpack Projected docs
 * (https://developer.android.com/develop/xr/jetpack-xr-sdk) but the exact
 * API names in Developer Preview 4 must be verified; spots are marked TODO.
 *
 * App model: this is an ordinary Android phone app. Its projected activity
 * reaches the glasses' outward camera and audio through a projected context.
 *
 * Manifest: <uses-permission android:name="android.permission.CAMERA"/>
 * plus the projected activity declaration described in the XR docs.
 *
 * NOT A MOBILITY AID. Never replace a white cane, guide dog or human help.
 */
package com.example.gazeassist

import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.Locale

// ---------- 1. Risk engine (ported from gaze_assist.py) ----------

data class Detection(val label: String, val centerX: Float, val distanceM: Float)

enum class Zone(val phrase: String) { LEFT("on your left"), AHEAD("ahead"), RIGHT("on your right") }

data class Hazard(
    val label: String, val zone: Zone, val distanceM: Float,
    val ttcS: Float?, val priority: Int, // 1 critical, 2 warning, 3 caution
)

class RiskEngine {
    private class Track(var d: Float, var tNs: Long, var v: Float)
    private val tracks = mutableMapOf<String, Track>() // prototype: real code should use IoU tracking

    fun assess(dets: List<Detection>, nowNs: Long): Hazard? =
        dets.map { assessOne(it, nowNs) }
            .filter { it.priority > 0 }
            .minWithOrNull(compareBy({ it.priority }, { it.distanceM }))

    private fun assessOne(d: Detection, nowNs: Long): Hazard {
        val tr = tracks.getOrPut(d.label) { Track(d.distanceM, nowNs, 0f) }
        val dt = (nowNs - tr.tNs) / 1e9f
        if (dt > 0.05f) {
            tr.v = 0.6f * tr.v + 0.4f * ((tr.d - d.distanceM) / dt) // closing speed, m/s
            tr.d = d.distanceM; tr.tNs = nowNs
        }
        val ttc = if (tr.v > 0.2f) d.distanceM / tr.v else null
        val zone = when {
            d.centerX < 0.38f -> Zone.LEFT
            d.centerX > 0.62f -> Zone.RIGHT
            else -> Zone.AHEAD
        }
        val p = when {
            d.distanceM <= 0.9f || (ttc != null && ttc <= 1.6f) ||
                (d.label == "stairs" && d.distanceM <= 1.6f) -> 1
            d.distanceM <= 1.8f || (ttc != null && ttc <= 3.0f) -> 2
            d.distanceM <= 3.0f -> 3
            else -> 0
        }
        return Hazard(d.label, zone, d.distanceM, ttc, p)
    }

    fun message(h: Hazard): String {
        val n = h.label.replaceFirstChar { it.uppercase() }
        val approaching = h.ttcS != null && h.ttcS <= 3f
        if (h.label == "stairs") return if (h.priority == 1) "Stairs very close ahead. Stop." else "Stairs ahead."
        if (h.priority == 2 && approaching) return "$n approaching ${h.zone.phrase}, about ${Math.round(h.ttcS!!)} seconds."
        val dist = String.format(Locale.US, "%.1f", h.distanceM)
        return "$n${if (approaching) " approaching" else ""} ${h.zone.phrase}, $dist meters." +
            if (h.priority == 1) " Stop." else ""
    }
}

// ---------- 2. Output: speech (audio glasses have no display) ----------

class AlertSpeaker(context: Context) {
    private var ready = false
    private val tts = TextToSpeech(context) { ready = it == TextToSpeech.SUCCESS; if (ready) tts.language = Locale.US }
    private var lastCriticalNs = 0L
    private var lastGeneralNs = 0L
    private val lastByKey = mutableMapOf<String, Long>()

    fun maybeSpeak(h: Hazard?, text: String, nowNs: Long) {
        if (h == null || !ready) return
        val key = h.label + h.zone
        if (h.priority == 1) {
            if (nowNs - lastCriticalNs < 2_500_000_000L) return
            lastCriticalNs = nowNs
        } else {
            if (nowNs - lastGeneralNs < 2_000_000_000L) return
            if (nowNs - (lastByKey[key] ?: 0L) < 5_000_000_000L) return
            lastGeneralNs = nowNs
        }
        lastByKey[key] = nowNs
        val mode = if (h.priority == 1) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        tts.speak(text, mode, null, key)
    }
}

// ---------- 3. Detector contract (plug in LiteRT / ML Kit here) ----------

interface SceneDetector {
    /** Return detections with centerX in [0,1] and an estimated distance in metres. */
    fun detect(frame: ImageProxy): List<Detection>
}

// ---------- 4. Glasses camera binding ----------

class GlassesAssist(
    private val glassesContext: Context, // TODO: obtain via createProjectedDeviceContext(...) per the XR docs
    private val owner: LifecycleOwner,
    private val detector: SceneDetector,
) {
    private val engine = RiskEngine()
    private val speaker = AlertSpeaker(glassesContext)

    fun start() {
        val future = ProcessCameraProvider.getInstance(glassesContext)
        future.addListener({
            val provider = future.get()
            // Newest-frame-only, mirroring the Python CaptureWorker.
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(ContextCompat.getMainExecutor(glassesContext)) { frame ->
                val now = System.nanoTime()
                val hazard = engine.assess(detector.detect(frame), now)
                if (hazard != null) speaker.maybeSpeak(hazard, engine.message(hazard), now)
                frame.close()
            }
            // Within a projected context the glasses' outward camera is DEFAULT_BACK_CAMERA.
            // The docs state the glasses camera supports one active stream, so bind
            // only ImageAnalysis (no Preview). TODO: verify against Developer Preview 4.
            provider.unbindAll()
            provider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(glassesContext))
    }
}
