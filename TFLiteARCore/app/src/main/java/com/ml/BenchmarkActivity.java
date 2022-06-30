package com.ml;

import androidx.appcompat.app.AppCompatActivity;

import com.common.Logger;
import com.ml.objectdetection.Classifier;
import com.ml.objectdetection.Coco;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.R;
import com.ml.objectdetection.ImageUtils;
import com.ml.objectdetection.Utils;
import com.ml.objectdetection.YoloV4Classifier;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class BenchmarkActivity extends AppCompatActivity {
    private static final Logger LOGGER = new Logger();
    private static final String COCO_FILE = "cocoVal.txt";
    public static final int TF_OD_API_INPUT_SIZE = 416;
    public static final String TF_OD_API_MODEL_FILE = "yolov4-416-fp16.tflite";
    public static final String TF_OD_API_LABELS_FILE = "coco.txt";
    public static final boolean TF_OD_API_IS_QUANTIZED = true;
    Handler handler = new Handler();
    private Bitmap sourceBitmap;
    private Bitmap cropBitmap;
    Coco coco;
    ImageView imageView;
    TextView preProcessTimeView, inferenceTimeView, postProcessTimeView, progressView, tpView, fpView, fnView, apView, arView, preProcessTimeViewAvg, inferenceTimeViewAvg, postProcessTimeViewAvg;
    Classifier detector;
    Matrix frameToCropTransform;
    int currentIndex = 0;
    long startTime = 0;
    long preProcessTime = 0;
    long inferenceTime = 0;
    long posProcessTime = 0;
    long preProcessTimeAvg = 0;
    long inferenceTimeAvg = 0;
    long posProcessTimeAvg = 0;
    int dataSetSize = 0;
    float recallScore = 0;
    float precisionScore = 0;
    int TP = 0;
    int FP = 0;
    int FN = 0;
    Runnable benchmarkRunnable = new Runnable() {
        @Override
        public void run() {
            try {
                benchmark();
            }catch(InterruptedException e){

            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_benchmark);
        imageView = findViewById(R.id.imageView);
        preProcessTimeView = findViewById(R.id.preProcessTime);
        inferenceTimeView = findViewById(R.id.inferenceTime);
        postProcessTimeView = findViewById(R.id.postProcessTime);
        preProcessTimeViewAvg = findViewById(R.id.preProcessTimeAvg);
        inferenceTimeViewAvg = findViewById(R.id.inferenceTimeAvg);
        postProcessTimeViewAvg = findViewById(R.id.postProcessTimeAvg);
        progressView = findViewById(R.id.progress);
        tpView = findViewById(R.id.TP);
        fpView = findViewById(R.id.FP);
        fnView = findViewById(R.id.FN);
        apView = findViewById(R.id.AP);
        arView = findViewById(R.id.AR);
        try{
            coco = new Coco(getAssets(),COCO_FILE);
            dataSetSize = coco.images.size();
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing coco dataset!");
            Toast toast = Toast.makeText(getApplicationContext(), "coco dataset could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
        try {
            detector = YoloV4Classifier.create(getAssets(),TF_OD_API_MODEL_FILE,TF_OD_API_LABELS_FILE,TF_OD_API_IS_QUANTIZED);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast = Toast.makeText(getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

       new Thread(benchmarkRunnable).start();
    }

    private void preProcess(){
        this.sourceBitmap = Utils.getBitmapFromAsset(BenchmarkActivity.this, coco.images.get(currentIndex).path+".jpg");
        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
        frameToCropTransform = ImageUtils.getTransformationMatrix(sourceBitmap.getWidth(), sourceBitmap.getHeight(),TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, 0, false);
    }

    private void benchmark() throws InterruptedException {
        while(currentIndex<coco.images.size()){
            preProcess();
            startTime = System.currentTimeMillis();
            ByteBuffer inputBuffer = detector.preProcess(sourceBitmap);
            preProcessTime = System.currentTimeMillis() - startTime;
            preProcessTimeAvg+=preProcessTime;
            startTime = System.currentTimeMillis();
            Map<Integer, Object> modelOutput = detector.runInference(inputBuffer);
            inferenceTime = System.currentTimeMillis() - startTime;
            inferenceTimeAvg+=inferenceTime;
            startTime = System.currentTimeMillis();
            List<Classifier.Recognition> results = detector.postProcess(modelOutput,cropBitmap);
            posProcessTime = System.currentTimeMillis() - startTime;
            posProcessTimeAvg+=posProcessTime;
            handleResult(cropBitmap, results);
            calculateMAP(coco.images.get(currentIndex).labels,results);
            Thread.sleep(1);
            currentIndex++;
        }
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= detector.getObjThresh()) {
                canvas.drawRect(location, paint);
//                cropToFrameTransform.mapRect(location);
//
//                result.setLocation(location);
//                mappedRecognitions.add(result);
            }
        }
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);
        for(Coco.Image.Label label:coco.images.get(currentIndex).labels){
            frameToCropTransform.mapRect(label.rect);
            canvas.drawRect(label.rect, paint);
        }
        handler.post(new Runnable(){
            public void run() {
                imageView.setImageBitmap(bitmap);
                preProcessTimeView.setText(String.valueOf(preProcessTime)+"ms");
                inferenceTimeView.setText(String.valueOf(inferenceTime)+"ms");
                postProcessTimeView.setText(String.valueOf(posProcessTime)+"ms");
                preProcessTimeViewAvg.setText(String.valueOf(preProcessTimeAvg/currentIndex)+"ms");
                inferenceTimeViewAvg.setText(String.valueOf(inferenceTimeAvg/currentIndex)+"ms");
                postProcessTimeViewAvg.setText(String.valueOf(posProcessTimeAvg/currentIndex)+"ms");
                progressView.setText(currentIndex+"/"+dataSetSize);
                if(currentIndex!=0) {
                    apView.setText(precisionScore / currentIndex + "");
                    arView.setText(recallScore / currentIndex + "");
                }
                tpView.setText(TP+"");
                fpView.setText(FP+"");
                fnView.setText(FN+"");
            }
        });
    }

    private void calculateMAP(List<Coco.Image.Label> labels, List<Classifier.Recognition> results){
        int precision = 0;
        int recall = 0;
        HashMap<Coco.Image.Label,Classifier.Recognition> tpMap = new HashMap<>();
        TP = 0;
        FP = 0;
        FN = 0;

        List<Classifier.Recognition> results2 = new ArrayList<>(results);
        List<Classifier.Recognition> resultsToRemove = new ArrayList<>();
        for(Classifier.Recognition result:results) {
            for (Classifier.Recognition result2 : results2) {
                if (result!=result2&&result2.getDetectedClass() == result.getDetectedClass() && detector.box_iou(result.getLocation(), result2.getLocation()) > 0.9) {
                    resultsToRemove.add(result);
                }
            }
        }
        results.removeAll(resultsToRemove);

        for(Classifier.Recognition result:results) {
            for(Coco.Image.Label label:labels){
                if(result.getDetectedClass()==label.classIndex){
                    if(detector.box_iou(result.getLocation(),label.rect)>0.5 && !tpMap.containsKey(label)){
                        tpMap.put(label,result);
                        TP++;
                    }
                }
            }
            if(!tpMap.containsValue(result)){
                FP++;
            }
            precision = TP/labels.size();
        }
        for(Coco.Image.Label label:labels){
            if(!tpMap.containsKey(label)){
                FN++;
            }
        }
        recallScore+=TP/(TP+FN);
        if(TP+FP!=0)
            precisionScore+=TP/(TP+FP);
    }
}