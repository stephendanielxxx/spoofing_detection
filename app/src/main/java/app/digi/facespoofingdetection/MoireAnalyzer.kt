package com.example.moiredetection

import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.core.CvType
import org.opencv.core.Core
import java.io.ByteArrayOutputStream
import android.graphics.Rect

@ExperimentalGetImage class MoireAnalyzer : ImageAnalysis.Analyzer {
    override fun analyze(imageProxy: ImageProxy) {
        val image = imageProxy.image ?: return
        val bitmap = toBitmap(image)

        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

//        isPossiblySpoofedFace(mat)

        detectScreenLikeObject(mat)

//        val gray = Mat()
//        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGBA2GRAY)
//
//        val highPass = Mat()
//        Imgproc.Laplacian(gray, highPass, CvType.CV_8U)
//
//        val mean = Core.mean(highPass).`val`[0]
//        if (mean > 75) {
//            Log.d("MoireDetection", "⚠️ FFFFFFFF Potential moiré or glare detected. Mean: $mean")
//        }else{
//            Log.d("MoireDetection", "⚠️ FFFFFFFF Real Face. Mean: $mean")
//        }
//        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)
//
//        val dftMat = Mat()
//        mat.convertTo(mat, CvType.CV_32F)
//        Core.dft(mat, dftMat, Core.DFT_COMPLEX_OUTPUT)
//
//        val planes = mutableListOf<Mat>()
//        Core.split(dftMat, planes)
//        val magnitude = Mat()
//        Core.magnitude(planes[0], planes[1], magnitude)
//
//        Core.normalize(magnitude, magnitude, 0.0, 255.0, Core.NORM_MINMAX)
//        val mean = Core.mean(magnitude).`val`[0]
//        val scaledMean = mean * 255
//        if (scaledMean > 75) {
//            Log.d("MoireDetection", "⚠️ FFFFFFFF Potential moiré or glare detected. Mean: $scaledMean")
//        }else{
//            Log.d("MoireDetection", "⚠️ FFFFFFFF Real Face. Mean: $scaledMean")
//        }

        imageProxy.close()
    }

    fun isPossiblySpoofedFace(frame: Mat): Boolean {
        // Step 1: Convert to Grayscale
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY)

        // Step 2: Apply Laplacian to detect texture edges
        val highPass = Mat()
        Imgproc.Laplacian(gray, highPass, CvType.CV_8U)

        // Step 3: Calculate mean intensity of high-pass result
        val mean = Core.mean(highPass).`val`[0]

        Log.d("SpoofCheck", "Laplacian mean: $mean")

        // Step 4: Decision threshold
        return when {
            mean < 10 -> {
                Log.w("SpoofCheck", "FFFFFFFF Too smooth — possible spoof (e.g. photo) $mean")
                true
            }
            mean > 60 -> {
                Log.w("SpoofCheck", "FFFFFFFF Too sharp — possible glare or screen spoof $mean")
                true
            }
            else -> {
                Log.i("SpoofCheck", "FFFFFFFF Texture normal — likely real $mean")
                false
            }
        }
    }

    fun detectScreenLikeObject(frame: Mat): Boolean {
        val gray = Mat()
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_RGBA2GRAY)

        // Edge detection
        val edges = Mat()
        Imgproc.Canny(gray, edges, 100.0, 200.0)

        // Count white pixels (edge density)
        val edgePixels = Core.countNonZero(edges)
        val totalPixels = gray.rows() * gray.cols()
        val edgeRatio = edgePixels.toDouble() / totalPixels

        // Glare detection using Laplacian
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        val mean = Core.mean(lap).`val`[0]

        Log.d("FFFFFFFF ScreenDetection", "EdgeRatio: $edgeRatio, Laplacian Mean: $mean")

        // Heuristic: if edgeRatio is very high and glare is strong, assume screen
        return edgeRatio > 0.05 //&& mean > 30
    }


    private fun toBitmap(image: Image): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val yuv = out.toByteArray()
        return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    }
}
