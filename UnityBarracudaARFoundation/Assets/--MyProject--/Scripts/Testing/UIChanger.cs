using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class UIChanger : MonoBehaviour
{
    public RawImage Image;
    public Text textRot;
    public Text textScale;
    public Slider sliderRot;
    public Slider sliderScale;
    public void Start(){
        sliderRot.onValueChanged.AddListener((v)=>{
            textRot.text = sliderRot.value.ToString();
            Image.rectTransform.localEulerAngles = new Vector3(0,0,v);
        });
        sliderScale.onValueChanged.AddListener((v)=>{
            textScale.text = sliderScale.value.ToString();
            Image.rectTransform.localScale = new Vector3(1f,v,1f);
        });
    }
    public void Update(){
    }
}
