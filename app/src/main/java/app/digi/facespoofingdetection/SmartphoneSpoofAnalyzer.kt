package app.digi.facespoofingdetection

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class SmartphoneSpoofAnalyzer(
    private val onResult: (isSpoofed: Boolean, lapMean: Double, edgeRatio: Double) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        val frame = Mat()
        Utils.bitmapToMat(bitmap, frame)
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)

        val result = detectSmartphoneSpoof(frame)
        onResult(result.first, result.second, result.third)

        image.close()
    }

    private fun detectSmartphoneSpoof(frame: Mat): Triple<Boolean, Double, Double> {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY)

        // 1. Edge detection
        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)

        val edgePixels = Core.countNonZero(edges)
        val totalPixels = gray.rows() * gray.cols()
        val edgeRatio = edgePixels.toDouble() / totalPixels

        // 2. Glare detection using Laplacian
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val mean = Core.mean(laplacian).`val`[0]

        val isSpoofed = edgeRatio > 0.05 && mean > 30

        Log.d("Analyzer", "Laplacian Mean: $mean, Edge Ratio: $edgeRatio, Spoofed: $isSpoofed")
        return Triple(isSpoofed, mean, edgeRatio)
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
