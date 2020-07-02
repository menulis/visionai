import com.google.api.core.ApiFutureCallback
import com.google.api.core.ApiFutures
import com.google.api.gax.longrunning.OperationFuture
import com.google.cloud.vision.v1.*
import com.google.cloud.vision.v1.Feature.Type
import java.io.File
import java.nio.file.Files
import java.nio.file.Files.isRegularFile
import java.nio.file.Paths
import java.util.concurrent.Executors

const val path = "/kelias/iki/partizanų/archyvo"
const val bucket = "gs://partizanų-dokumentų-tekstai"

fun main(args: Array<String>) {
    ImageAnnotatorClient.create().use { vision ->
        val operationFutures =
            mutableListOf<OperationFuture<AsyncBatchAnnotateImagesResponse, OperationMetadata>>()
        Files.list(Paths.get(path))
            .forEach { dir ->
                val requestElements = mutableListOf<AnnotateImageRequest>()
                val outputUri = "$bucket/${dir.last()}/"
                val gcsDestination = GcsDestination.newBuilder().setUri(outputUri).build()
                val batchSize = 1
                val outputConfig = OutputConfig.newBuilder()
                    .setGcsDestination(gcsDestination)
                    .setBatchSize(batchSize)
                    .build()

                Files.walk(dir)
                    .filter { f -> isRegularFile(f) && File(f.toString()).extension.toLowerCase() == "jpg" }
                    .forEach { f ->
                        val inputImageUri = "$bucket/${Paths.get(path).relativize(f)}"
                        val source = ImageSource.newBuilder().setImageUri(inputImageUri).build()
                        val feat = Feature.newBuilder().setType(Type.DOCUMENT_TEXT_DETECTION).build()
                        val requestsElement = AnnotateImageRequest.newBuilder()
                            .addFeatures(feat)
                            .setImage(Image.newBuilder().setSource(source).build())
                            // .setImageContext(ImageContext.newBuilder().addLanguageHints("lt"))
                            .build()

                        requestElements.add(requestsElement)
                    }

                if (requestElements.isEmpty()) return@forEach

                val request = AsyncBatchAnnotateImagesRequest.newBuilder()
                    .addAllRequests(requestElements)
                    .setOutputConfig(outputConfig)
                    .build()

                println("${dir.last()}: ${requestElements.size} requests")

                val operationFuture =
                    vision.asyncBatchAnnotateImagesAsync(request)
                ApiFutures.addCallback(
                    operationFuture,
                    object : ApiFutureCallback<AsyncBatchAnnotateImagesResponse> {
                        override fun onFailure(t: Throwable?) {
                            println("onFailure: $t")
                        }

                        override fun onSuccess(result: AsyncBatchAnnotateImagesResponse?) {
                            println("onSuccess: $result")
                            // val gcsOutputUri = response.outputConfig.gcsDestination.uri
                        }

                    },
                    Executors.newCachedThreadPool()
                )
                operationFutures.add(operationFuture)

            }

        println("Waiting for ${operationFutures.size} operations to complete")
        ApiFutures.allAsList(operationFutures).get().let {
           println("Completed waiting for ${operationFutures.size} operations, done: ${it.size}")
        }

    }
}
