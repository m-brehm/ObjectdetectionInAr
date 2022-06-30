using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using YoloV4Tiny;
using Unity.Barracuda;
[CustomEditor(typeof(ComputeShaderTest))]
public class ComputeShaderTestInspector : Editor
{
    public override void OnInspectorGUI(){
        base.OnInspectorGUI();
        if(GUILayout.Button("Test")){
            var settings = serializedObject.targetObject as ComputeShaderTest;
            var compute = settings.compute;
            int size = settings.size;
            RenderTexture OutputBuffer = new RenderTexture(size, size, 0);
            settings.Source.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal,settings.Source.texture.width);
            settings.Source.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,settings.Source.texture.height);
            ComputeBuffer buffer = new ComputeBuffer(size*size*3, sizeof(float));
            //var tex = new Texture2D(_config.InputWidth,_config.InputWidth);
            var tex = new RenderTexture(size, size, 0, RenderTextureFormat.RFloat);
            tex.enableRandomWrite = true;
            Graphics.Blit(settings.Source.texture,OutputBuffer);
            compute.SetInt("Size", size);
            compute.SetTexture(0, "Image", OutputBuffer);
            compute.SetTexture(0,"Result",tex);
            compute.SetBuffer(0, "Tensor", buffer);
            int kernel = 0;
            int x = size;
            int y = size;
            int z = 1;
            uint xc, yc, zc;
            compute.GetKernelThreadGroupSizes(kernel, out xc, out yc, out zc);

            x = (x + (int)xc - 1) / (int)xc;
            y = (y + (int)yc - 1) / (int)yc;
            z = (z + (int)zc - 1) / (int)zc;

            compute.Dispatch(kernel, x, y, z);
            /*
            settings.Result.texture = tex;
            settings.Result.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal,settings.Result.texture.width);
            settings.Result.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,settings.Result.texture.height);
            */
            var t = new Tensor(new TensorShape(1, size, size, 3), buffer);
            var pixels = new Color32[size*size];
            for(int yi=0;yi<size;yi++){
                for(int xi=0;xi<size;xi++){
                    UnityEngine.Color sourceColor = new UnityEngine.Color(t[0,yi,xi,0], t[0,yi,xi,1], t[0,yi,xi,2], 1f);
                    pixels[yi*size+xi]=sourceColor;
                }
            }
            var texT = new Texture2D(size,size);
            texT.SetPixels32(pixels);
            texT.Apply();
            //Graphics.Blit(texT,OutputBuffer);
            settings.Result.texture = texT;
            settings.Result.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal,settings.Result.texture.width);
            settings.Result.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,settings.Result.texture.height);
        }

    }
}
