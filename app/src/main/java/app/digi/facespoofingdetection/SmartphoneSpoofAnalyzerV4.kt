package app.digi.facespoofingdetection

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream

class SmartphoneSpoofAnalyzerV4(
    private val onResult: (isSpoofed: Boolean) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val bitmap = imageProxyToBitmap(image)
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB)

        val isSpoof = detectSpoofStrong(mat)
        onResult(isSpoof)
        image.close()
    }

    private fun detectSpoofStrong(frame: Mat): Boolean {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGB2GRAY)
        val totalPixels = gray.rows() * gray.cols()

        // High edge density
        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)
        val edgeDensity = Core.countNonZero(edges).toDouble() / totalPixels

        // Low texture (Laplacian)
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(lap, mean, stddev)

        val lapVar = stddev[0, 0][0] // standard deviation of Laplacian
        Log.d("Spoof", "Laplacian stddev: $lapVar")

        // YCbCr check
        val ycrcb = Mat()
        Imgproc.cvtColor(frame, ycrcb, Imgproc.COLOR_RGB2YCrCb)
        val skinMask = Mat()
        Core.inRange(ycrcb, Scalar(0.0, 133.0, 77.0), Scalar(255.0, 173.0, 127.0), skinMask)
        val skinRatio = Core.countNonZero(skinMask).toDouble() / totalPixels

        // Bright spot (glare)
        val bright = Mat()
        Imgproc.threshold(gray, bright, 240.0, 255.0, Imgproc.THRESH_BINARY)
        val glareRatio = Core.countNonZero(bright).toDouble() / totalPixels

        Log.d("Spoof", "EdgeDensity=$edgeDensity LapVar=$lapVar Skin=$skinRatio Glare=$glareRatio")

        // Combine all features
        val score = listOf(
            if (edgeDensity > 0.08) 1 else 0,       // Sharp edges
            if (lapVar < 10) 1 else 0,              // Flat texture
            if (skinRatio < 0.15) 1 else 0,         // Low skin tone
            if (glareRatio > 0.005) 1 else 0        // Strong reflection
        ).sum()

        return score >= 2 // Tweak threshold for stricter detection
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
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }
}
