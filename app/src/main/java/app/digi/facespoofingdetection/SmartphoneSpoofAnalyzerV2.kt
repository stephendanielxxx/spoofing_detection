package app.digi.facespoofingdetection

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class SmartphoneSpoofAnalyzerV2(
    private val onResult: (isSpoofed: Boolean) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        val frame = Mat()
        Utils.bitmapToMat(bitmap, frame)
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)

        val result = detectSpoof(frame)
        onResult(result)

        image.close()
    }

    private fun detectSpoof(frame: Mat): Boolean {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY)
        val totalPixels = gray.rows() * gray.cols()

        // Edge detection
        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)
        val edgePixels = Core.countNonZero(edges)
        val edgeRatio = edgePixels.toDouble() / totalPixels
        Log.d("Spoof", "Edge ratio: $edgeRatio")

        // Laplacian for texture
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val lapMean = Core.mean(lap).`val`[0]
        Log.d("Spoof", "Laplacian mean: $lapMean")

        // Brightness / glare
        val bright = Mat()
        Imgproc.threshold(gray, bright, 240.0, 255.0, Imgproc.THRESH_BINARY)
        val brightPixels = Core.countNonZero(bright)
        val brightRatio = brightPixels.toDouble() / totalPixels
        Log.d("Spoof", "Bright pixel ratio: $brightRatio")

        // Moiré detection using Sobel
        val sobelX = Mat()
        val sobelY = Mat()
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_16S, 0, 1)
        val sobel = Mat()
        Core.addWeighted(sobelX, 0.5, sobelY, 0.5, 0.0, sobel)
        val sobelMean = Core.mean(sobel).`val`[0]
        Log.d("Spoof", "Sobel mean: $sobelMean")

        // Rectangle detection (detect phone shape)
        var foundRect = false
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        for (c in contours) {
            val approx = MatOfPoint2f()
            val contour2f = MatOfPoint2f(*c.toArray())
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * Imgproc.arcLength(contour2f, true), true)

            if (approx.total() == 4L && Imgproc.contourArea(c) > 10000) {
                foundRect = true
                break
            }
        }
        Log.d("Spoof", "Rectangle detected: $foundRect")

        val isSpoof = (edgeRatio > 0.05 && lapMean > 30) ||
                (brightRatio > 0.01) ||
                (sobelMean > 20) ||
                foundRect

        return isSpoof
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

        val yuvImage = android.graphics.YuvImage(
            nv21, android.graphics.ImageFormat.NV21,
            imageProxy.width, imageProxy.height, null
        )

        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }
}
