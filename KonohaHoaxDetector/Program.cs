using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace KonohaHoaxDetector
{
    public class Program
    {
        // Class untuk menyimpan data berita
        public class NewsData
        {
            [LoadColumn(0)]
            public string Label { get; set; } = string.Empty;

            [LoadColumn(1)]
            public string Berita { get; set; } = string.Empty;
        }

        // Class untuk hasil prediksi
        public class NewsPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabel { get; set; } = string.Empty;
        }

        static void Main(string[] args)
        {
            // 1️ Inisialisasi MLContext
            var mlContext = new MLContext();

            // 2️ Memuat Dataset
            Console.WriteLine("1️. Memuat Dataset dan Membagi Data...");
            string dataPath = "news_hoax_dataset_v2.csv";

            var fullData = mlContext.Data.LoadFromTextFile<NewsData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            Console.WriteLine($"   Total Data: {mlContext.Data.CreateEnumerable<NewsData>(fullData, reuseRowObject: false).Count()}");
            Console.WriteLine($"   Data Training: {mlContext.Data.CreateEnumerable<NewsData>(trainData, reuseRowObject: false).Count()}");

            // 3️ Definisikan Pipeline
            Console.WriteLine("2️. Mendefinisikan Pipeline ML.NET (Tokenize + Normalize + TF-IDF + Trainer)...");
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Label", outputColumnName: "KeyedLabel")
                .Append(mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Berita", outputColumnName: "Features"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "KeyedLabel", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    inputColumnName: "PredictedLabel", outputColumnName: "PredictedLabel"));

            // 4️ Melatih Model
            Console.WriteLine("3️. Melatih Model...");
            var model = pipeline.Fit(trainData);

            // 5️ Evaluasi Model
            Console.WriteLine("4️. Mengevaluasi Model...");
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "KeyedLabel", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"\nAkurasi Mikro: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"Akurasi Makro: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");
            Console.WriteLine("\nModel telah selesai dilatih dan dievaluasi\n");

            // 6️ Simpan Model
            string modelPath = "model_hoax.zip";
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"Model disimpan ke: {modelPath}\n");

            // 7️ Contoh Prediksi
            Console.WriteLine("5. Contoh Prediksi Berita Baru...");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<NewsData, NewsPrediction>(model);

            var sample1 = new NewsData
            {
                Berita = "Pemerintah telah menyelesaikan pembangunan infrastruktur jalan tol di kawasan Bandung"
            };

            var sample2 = new NewsData
            {
                Berita = "Semua warga wajib membayar denda 10 juta atau akan dipenjara"
            };

            var result1 = predictionEngine.Predict(sample1);
            var result2 = predictionEngine.Predict(sample2);

            Console.WriteLine($"Berita 1: {sample1.Berita}");
            Console.WriteLine($"   Prediksi: {result1.PredictedLabel}");

            Console.WriteLine($"\nBerita 2: {sample2.Berita}");
            Console.WriteLine($"   Prediksi: {result2.PredictedLabel}");

            Console.WriteLine("\nSelesai");
        }
    }
}
