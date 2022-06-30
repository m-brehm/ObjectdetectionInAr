using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using UnityEngine.Video;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using System;
using ObjectDetection;
using System.IO;


public sealed class ImageSource : MonoBehaviour
{
    #region Public property

    public Texture Texture => OutputBuffer;
    public Texture2D Texture2D => Utils.PreProcessing.toTexture2D(OutputBuffer);

    #endregion

    #region Editable attributes

    // Source type options
    public enum SourceType { Texture, Video, Webcam, Card, Gradient, AR_Camera, COCO }
    [SerializeField] SourceType _sourceType = SourceType.Card;

    // Texture mode options
    [SerializeField] public Texture2D _texture = null;
    [SerializeField] string _textureUrl = null;

    // Video mode options
    [SerializeField] VideoClip _video = null;
    [SerializeField] string _videoUrl = null;

    // Webcam options
    [SerializeField] string _webcamName = "";
    [SerializeField] Vector2Int _webcamResolution = new Vector2Int(1920, 1080);
    [SerializeField] int _webcamFrameRate = 30;

    //AR options
    [SerializeField] ARCameraManager _arCamera;

    //COCO options
    [SerializeField] TextAsset _annotations;
 

    // Output options
    [SerializeField] RenderTexture _outputTexture = null;
    [SerializeField] public RawImage Preview = null;
    [SerializeField] Vector2Int _outputResolution = new Vector2Int(1920, 1080);


    #endregion

    #region Package asset reference

    [SerializeField, HideInInspector] Shader _shader = null;

    #endregion

    #region Private members

    UnityWebRequest _webTexture;
    WebCamTexture _webcam;
    Material _material;
    RenderTexture _buffer;
    Texture2D _arTexture;
    Camera _mainCamera;
    public Coco _dataset;
    public int currentImageIndex = 0;

    RenderTexture OutputBuffer
      => _outputTexture != null ? _outputTexture : _buffer;

    // Blit a texture into the output buffer with aspect ratio compensation.
    void Blit(Texture source, bool vflip = false)
    {
        if (source == null) return;

        var aspect1 = (float)source.width / source.height;
        var aspect2 = (float)OutputBuffer.width / OutputBuffer.height;
        var gap = aspect2 / aspect1;

        var scale = new Vector2(gap, vflip ? -1 : 1);
        var offset = new Vector2((1 - gap) / 2, vflip ? 1 : 0);

        Graphics.Blit(source, OutputBuffer, scale, offset);
    }

    #endregion

    #region MonoBehaviour implementation

    void Start()
    {
        // Allocate a render texture if no output texture has been given.
        if (_outputTexture == null)
            _buffer = new RenderTexture
              (_outputResolution.x, _outputResolution.y, 0);

        // Create a material for the shader (only on Card and Gradient)
        if (_sourceType == SourceType.Card || _sourceType == SourceType.Gradient)
            _material = new Material(_shader);

        // Texture source type:
        // Blit a given texture, or download a texture from a given URL.
        if (_sourceType == SourceType.Texture)
        {
            if (_texture != null)
            {
                Graphics.Blit(_texture,OutputBuffer);
            }
            else
            {
                _webTexture = UnityWebRequestTexture.GetTexture(_textureUrl);
                _webTexture.SendWebRequest();
            }
        }

        // Video source type:
        // Add a video player component and play a given video clip with it.
        if (_sourceType == SourceType.Video)
        {
            var player = gameObject.AddComponent<VideoPlayer>();
            player.source =
              _video != null ? VideoSource.VideoClip : VideoSource.Url;
            player.clip = _video;
            player.url = _videoUrl;
            player.isLooping = true;
            player.renderMode = VideoRenderMode.APIOnly;
            player.Play();
        }

        // Webcam source type:
        // Create a WebCamTexture and start capturing.
        if (_sourceType == SourceType.Webcam)
        {
            _webcam = new WebCamTexture(WebCamTexture.devices[0].name,_webcamResolution.x, _webcamResolution.y, _webcamFrameRate);
            _webcam.Play();
            //float scaleY = _webcam.videoVerticallyMirrored ? -1f:1f;
            //Preview.rectTransform.localScale = new Vector3(1f,scaleY,1f);
            //Preview.rectTransform.localEulerAngles = new Vector3(0,0,-_webcam.videoRotationAngle);
        }

        if (_sourceType == SourceType.AR_Camera && _arCamera != null)
        {
            //_arCamera.frameReceived += OnCameraFrameReceived;
            _mainCamera = _arCamera.GetComponent<Camera>();
            _mainCamera.targetTexture = OutputBuffer;
        }
        // Card source type:
        // Run the card shader to generate a test card image.
        if (_sourceType == SourceType.Card)
        {
            var dims = new Vector2(OutputBuffer.width, OutputBuffer.height);
            _material.SetVector("_Resolution", dims);
            Graphics.Blit(null, OutputBuffer, _material, 0);
        }
        if(_sourceType == SourceType.COCO)
        {
            _dataset = new Coco(_annotations);
            _texture = Resources.Load(_dataset.images[currentImageIndex].path) as Texture2D;
            Graphics.Blit(_texture,OutputBuffer);
        }
    }

    void OnDestroy()
    {
        if (_webcam != null) Destroy(_webcam);
        if (_buffer != null) Destroy(_buffer);
        if (_material != null) Destroy(_material);
        if (_arCamera != null)_arCamera.frameReceived -= OnCameraFrameReceived;
    }

    void Update()
    {
        if (_sourceType == SourceType.Video){
            Graphics.Blit(GetComponent<VideoPlayer>().texture,OutputBuffer);
        }

        if (_sourceType == SourceType.Webcam && _webcam.didUpdateThisFrame){
            Graphics.Blit(_webcam, OutputBuffer);
        }

        // Asynchronous image downloading
        if (_webTexture != null && _webTexture.isDone)
        {
            var texture = DownloadHandlerTexture.GetContent(_webTexture);
            _webTexture.Dispose();
            _webTexture = null;
            Graphics.Blit(texture, OutputBuffer);
            Destroy(texture);
        }

        if (_sourceType == SourceType.Gradient){
            Graphics.Blit(null, OutputBuffer, _material, 1);
        }
        if (_sourceType == SourceType.AR_Camera){
            //_mainCamera.Render();
            //_arCamera.GetComponent<Camera>().targetTexture = OutputBuffer;
            //Graphics.Blit(null, OutputBuffer, _material, 1);
        }
        if (_sourceType == SourceType.COCO){
            //nextImage();
        }
        Preview.texture = Texture;
    }

    public bool nextImage(){
        if(currentImageIndex < _dataset.images.Count-1){
            currentImageIndex++;
            if(Resources.Load(_dataset.images[currentImageIndex].path)!=null)
                _texture = Resources.Load(_dataset.images[currentImageIndex].path) as Texture2D;
            Graphics.Blit(_texture,OutputBuffer);
            Preview.texture = Texture;
            //Preview.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal,416.0f);//Preview.texture.width);
            //Preview.rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical,416.0f);//Preview.texture.height);
            if(currentImageIndex%100==0)
                Resources.UnloadUnusedAssets();
            return true;
        }
        return false;
    }

    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        Graphics.Blit(GetCurrentColorTexture(),OutputBuffer);
    }

    unsafe Texture2D GetCurrentColorTexture()
    {
        if (!_arCamera.TryAcquireLatestCpuImage(out XRCpuImage image))
            return null;


        XRCpuImage.ConversionParams conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(image.width, image.height),
            outputFormat = TextureFormat.RGBA32,
            transformation = XRCpuImage.Transformation.MirrorX
        };

        if (_arTexture == null || _arTexture.width != image.width || _arTexture.height != image.height)
        {
            _arTexture = new Texture2D(conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, conversionParams.outputFormat, false);
        }

        var rawTextureData = _arTexture.GetRawTextureData<byte>();

        image.Convert(conversionParams, new IntPtr(rawTextureData.GetUnsafePtr()), rawTextureData.Length);
        image.Dispose();


        _arTexture.Apply();
        return _arTexture;
    }

    #endregion
}

