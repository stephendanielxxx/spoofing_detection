package app.digi.facespoofingdetection

import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import android.graphics.Rect

class SmartphoneSpoofAnalyzerV3(
    private val onResult: (isSpoofed: Boolean) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        val frame = Mat()
        Utils.bitmapToMat(bitmap, frame)

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)

        val isSpoof = detectSpoof(frame)
        onResult(isSpoof)

        image.close()
    }

    private fun detectSpoof(frame: Mat): Boolean {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY)

        val totalPixels = gray.rows() * gray.cols()

        // ---- Texture: Laplacian Variance ----
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val lapMean = Core.mean(lap).`val`[0]
        Log.d("Spoof", "Laplacian mean: $lapMean")

        // ---- Glare: Brightness ratio ----
        val bright = Mat()
        Imgproc.threshold(gray, bright, 240.0, 255.0, Imgproc.THRESH_BINARY)
        val brightRatio = Core.countNonZero(bright).toDouble() / totalPixels
        Log.d("Spoof", "Bright pixel ratio: $brightRatio")

        // ---- Moiré/Sobel ----
        val sobelX = Mat()
        val sobelY = Mat()
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)
        val sobel = Mat()
        Core.addWeighted(sobelX, 0.5, sobelY, 0.5, 0.0, sobel)
        val sobelMean = Core.mean(sobel).`val`[0]
        Log.d("Spoof", "Sobel mean: $sobelMean")

        // ---- Rectangle detection ----
        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        var hasRectangle = false
        for (contour in contours) {
            val approx = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*contour.toArray())
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * Imgproc.arcLength(contour2f, true), true)
            if (approx.total() == 4L && Imgproc.contourArea(contour) > 8000) {
                hasRectangle = true
                break
            }
        }
        Log.d("Spoof", "Rectangle detected: $hasRectangle")

        // ---- Optional: Skin detection (basic YCbCr filter) ----
        val ycrcb = Mat()
        Imgproc.cvtColor(frame, ycrcb, Imgproc.COLOR_RGB2YCrCb)
        val skinMask = Mat()
        Core.inRange(ycrcb, Scalar(0.0, 133.0, 77.0), Scalar(255.0, 173.0, 127.0), skinMask)
        val skinRatio = Core.countNonZero(skinMask).toDouble() / totalPixels
        Log.d("Spoof", "Skin pixel ratio: $skinRatio")

        val spoofScore = listOf(
            if (lapMean < 15) 1 else 0,           // Very low texture
            if (brightRatio > 0.008) 1 else 0,    // Glare
            if (sobelMean > 10) 1 else 0,         // Moiré or LCD lines
            if (hasRectangle) 1 else 0,           // Likely phone shape
            if (skinRatio < 0.15) 1 else 0        // Skin not visible
        ).sum()

        Log.d("Spoof", "Spoof score: $spoofScore")
        return spoofScore >= 2  // You can increase to >=3 for more strictness
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
