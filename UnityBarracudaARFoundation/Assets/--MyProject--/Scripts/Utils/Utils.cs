using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using TFClassify;

namespace Utils
{
    #region Object construction/destruction helpers

    static class ObjectUtil
    {
        public static void Destroy(UnityEngine.Object o)
        {
            if (o == null) return;
            if (Application.isPlaying)
                UnityEngine.Object.Destroy(o);
            else
                UnityEngine.Object.DestroyImmediate(o);
        }
    }

    static class RTUtil
    {
        public static RenderTexture NewFloat(int w, int h)
          => new RenderTexture(w, h, 0, RenderTextureFormat.RFloat);
    }

    #endregion

    #region Extension methods

    static class ComputeShaderExtensions
    {
        public static void DispatchThreads
          (this ComputeShader compute, int kernel, int x, int y, int z)
        {
            uint xc, yc, zc;
            compute.GetKernelThreadGroupSizes(kernel, out xc, out yc, out zc);

            x = (x + (int)xc - 1) / (int)xc;
            y = (y + (int)yc - 1) / (int)yc;
            z = (z + (int)zc - 1) / (int)zc;

            compute.Dispatch(kernel, x, y, z);
        }
    }

    static class IWorkerExtensions
    {
        public static void CopyOutput
          (this IWorker worker, string tensorName, RenderTexture rt)
        {
            var output = worker.PeekOutput(tensorName);
            var shape = new TensorShape(1, rt.height, rt.width, 1);
            using var tensor = output.Reshape(shape);
            tensor.ToRenderTexture(rt);
            //Debug.Log("test "+output);
        }
    }

    #endregion

    #region GPU to CPU readback helpers

    sealed class DetectionCache
    {
        ComputeBuffer _dataBuffer;
        ComputeBuffer _countBuffer;

        Detection[] _cached;
        int[] _countRead = new int[1];

        public DetectionCache(ComputeBuffer data, ComputeBuffer count)
          => (_dataBuffer, _countBuffer) = (data, count);

        public Detection[] Cached => _cached ?? UpdateCache();

        public void Invalidate() => _cached = null;

        public Detection[] UpdateCache()
        {
            _countBuffer.GetData(_countRead, 0, 0, 1);
            var count = _countRead[0];

            _cached = new Detection[count];
            _dataBuffer.GetData(_cached, 0, 0, count);

            return _cached;
        }
    }

    #endregion
    public static class PreProcessing
    {
        private const int IMAGE_MEAN = 0;
        private const float IMAGE_STD = 255.0F;
        public static Tensor TransformInput(Color32[] pic, int width, int height)
        {
            float[] floatValues = new float[width * height * 3];

            for (int i = 0; i < pic.Length; ++i)
            {
                var color = pic[i];

                floatValues[i * 3 + 0] = (color.r - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 1] = (color.g - IMAGE_MEAN) / IMAGE_STD;
                floatValues[i * 3 + 2] = (color.b - IMAGE_MEAN) / IMAGE_STD;
            }

            return new Tensor(1, height, width, 3, floatValues);
        }

        public static Texture2D toTexture2D(RenderTexture rTex)
        {
            Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
            tex.Apply(false);
            Graphics.CopyTexture(rTex, tex);
            return tex;

            /*
            // ReadPixels looks at the active RenderTexture.
            RenderTexture.active = rTex;
            tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
            tex.Apply();
            return tex;
            */
        }
        public static Tensor ProcessImage(Texture2D m_Texture, int inputSize)
        {
            var scaled = Scale(m_Texture, inputSize);
            var rotated = Rotate(scaled.GetPixels32(), scaled.width, scaled.height);
            return TransformInput(rotated,inputSize,inputSize);
        }
        private static Texture2D Scale(Texture2D texture, int imageSize)
        {
            var scaled = TextureTools.scaled(texture, imageSize, imageSize, FilterMode.Bilinear);
            return scaled;
        }


        private static Color32[] Rotate(Color32[] pixels, int width, int height)
        {
            var rotate = TextureTools.RotateImageMatrix(
                    pixels, width, height, 90);
            // var flipped = TextureTools.FlipYImageMatrix(rotate, width, height);
            //flipped =  TextureTools.FlipXImageMatrix(flipped, width, height);
            // return flipped;
            return rotate;
        }
    }

    public static class PostProcessing
    {
        public static float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);

            return k / (1.0f + k);
        }

        public static float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        public static BoundingBoxDimensions ExtractBoundingBoxDimensions(Tensor modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[0, x, y, channel],
                Y = modelOutput[0, x, y, channel + 1],
                Width = modelOutput[0, x, y, channel + 2],
                Height = modelOutput[0, x, y, channel + 3]
            };
        }

        public static BoundingBoxDimensions ExtractBoundingBoxDimensions(Tensor modelOutput, int b)
        {
            return new BoundingBoxDimensions
            {
                X = (modelOutput[0, 0, 0, b] - modelOutput[0, 0, 2, b] / 2),
                Y = (modelOutput[0, 0, 1, b] - modelOutput[0, 0, 3, b] / 2),
                Width = (modelOutput[0, 0, 0, b] + modelOutput[0, 0, 2, b] / 2),
                Height = (modelOutput[0, 0, 1, b] + modelOutput[0, 0, 3, b] / 2)
            };
        }

        public static float GetConfidence(Tensor modelOutput, int x, int y, int channel)
        {
            //Debug.Log("ModelOutput " + modelOutput);
            return Sigmoid(modelOutput[0, x, y, channel + 4]);
        }

        public static float GetConfidence(Tensor modelOutput, int b)
        {
            //Debug.Log("ModelOutput " + modelOutput);
            //return Sigmoid(modelOutput[0, 0, 4, b]);
            return modelOutput[0, 0, 4, b];
        }

        public static CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions, int cellSize, int width, float[] anchors)
        {
            return new CellDimensions
            {
                X = ((float)y + Sigmoid(boxDimensions.X)) / cellSize,
                Y = ((float)x + Sigmoid(boxDimensions.Y)) / cellSize,
                Width = (float)Math.Exp(boxDimensions.Width) * (anchors[6 + box * 2]*(1.0f/width)),
                Height = (float)Math.Exp(boxDimensions.Height) * (anchors[6 + box * 2 + 1]*(1.0f/width)),
            };
        }

        public static CellDimensions MapBoundingBoxToCell(int b, BoundingBoxDimensions boxDimensions)
        {
            return new CellDimensions
            {
                X = boxDimensions.X,
                Y = boxDimensions.Y,
                Width = boxDimensions.Width - boxDimensions.X,
                Height = boxDimensions.Height - boxDimensions.Y
            };
        }

        public static float[] ExtractClasses(Tensor modelOutput, int x, int y, int channel, int CLASS_COUNT, int BOX_INFO_FEATURE_COUNT)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;

            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[0, x, y, predictedClass + predictedClassOffset];
            }

            return Softmax(predictedClasses);
        }

        public static float[] ExtractClasses(Tensor modelOutput, int b, int CLASS_COUNT, int BOX_INFO_FEATURE_COUNT)
        {
            float[] predictedClasses = new float[CLASS_COUNT];
            int predictedClassOffset = BOX_INFO_FEATURE_COUNT;

            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[0, 0, predictedClass + predictedClassOffset, b];
            }

            return predictedClasses;
        }
        public static float[] ExtractClasses(Tensor modelOutput, int b, int CLASS_COUNT)
        {
            float[] predictedClasses = new float[CLASS_COUNT];

            for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[0, 0, predictedClass, b];
            }

            return predictedClasses;
        }
        public static IList<Detection> FilterBoundingBoxes(IList<Detection> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
            {
                isActiveBoxes[i] = true;
            }

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.score)
                    .ToList();

            var results = new List<Detection>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (Measurements.IntersectionOverUnion(new Rect(boxA.x,boxA.y,boxA.w,boxA.h), new Rect(boxB.x,boxB.y,boxB.w,boxB.h)) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

        public static IList<BoundingBox> FilterBoundingBoxes(IList<BoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
            {
                isActiveBoxes[i] = true;
            }

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            var results = new List<BoundingBox>();

            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (Measurements.IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

        public static ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }
    }

    public static class Measurements
    {
        public static float IntersectionOverUnion(Rect boundingBoxA, Rect boundingBoxB)
        {
            var areaA = boundingBoxA.width * boundingBoxA.height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.width * boundingBoxB.height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.xMin, boundingBoxB.xMin);
            var minY = Math.Max(boundingBoxA.yMin, boundingBoxB.yMin);
            var maxX = Math.Min(boundingBoxA.xMax, boundingBoxB.xMax);
            var maxY = Math.Min(boundingBoxA.yMax, boundingBoxB.yMax);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }
    }

    public class DimensionsBase
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }

    public class BoundingBoxDimensions : DimensionsBase { }

    public class CellDimensions : DimensionsBase { }

    public class BoundingBox
    {
        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }

        // whether the bounding box already is used to raycast anchors
        public bool Used { get; set; }

        public Rect Rect
        {
            get { return new Rect(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
        }

        public override string ToString()
        {
            return $"{Label}:{Confidence}, {Dimensions.X}:{Dimensions.Y} - {Dimensions.Width}:{Dimensions.Height}";
        }
    }
}
