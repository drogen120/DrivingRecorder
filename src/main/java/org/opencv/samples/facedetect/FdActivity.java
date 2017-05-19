package org.opencv.samples.facedetect;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Calendar;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Criteria;
import android.location.GpsStatus;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.media.CamcorderProfile;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.Toast;


public class FdActivity extends Activity implements CvCameraViewListener2, LocationListener, SensorEventListener, View.OnClickListener, GpsStatus.NmeaListener {

    private static final String    TAG                 = "DrivingRecorder::Activity";
    private String recordfilePath;
    private String startTime;  //測定開始時刻（文字列）
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    private TextView gpsStateText, sensorStateText, accRateText,magRateText,gyroRateText, linearRateText, recodeStateText;
    private Button startstopButton;
    private CompoundButton videotoggle;
    private LocationManager locManager;
    private boolean isFixed; //GPSの位置情報が安定しているかどうか
    private static final float NS2S = 1.0f / 100000.0f;
    private float timestamp;
    private SensorManager sensorManager;
    private Sensor accelerometer, linearaccelerometer, magnetometer, gyroscope, pressure;

    private static final String COMMA = ",";
    private static final String NMEA_GPGSA = "$GPGSA";
    private static final String NMEA_GPGGA = "$GPGGA";
    private static final int GGA_UTC_COLUMN = 1;
    private static final int GSA_PDOP_COLUMN = 15;
    private static final int GSA_HDOP_COLUMN = 16;
    private static final int GSA_VDOP_COLUMN = 17;

    private String nmeaPdop, nmeaHdop, nmeaVdop,nmeaUTC;
    private static final float START_ACCURACY_THRESH = 150f; //この精度でGPSの位置情報が取れたら測定開始できる[m]
    private static final long GPS_INTERVAL = 1000; //GPS測定の最短間隔[ミリ秒]
    private static final int SENSOR_RATE = 20;
    private static final int SENSOR_INTERVAL = 1000000/SENSOR_RATE; //加速度／磁気測定の（最短）間隔[マイクロ秒]
    private long accelRefTime; //加速度サンプリングレート検査用
    private long gyroRefTime;
    private long linearRefTime;
    private int accelCount, magCount, gyroCount, linearCount;    //同上
    private int ignore_count;
    private long uptime_nano; //スマホの電源が入ってからの時間。センサ類のtimestampの値が保存される変数。
    private long starttimestamp;
    private boolean videoRecode = true;
    private boolean logStarted; //測定中かどうか
    private PrintWriter gpsWriter, imglistWriter;
    private File gpsFile, imglistFile;
    private String SDCardPath; //一時ファイルや結果ファイルの保存ディレクトリのパス
    private boolean recordStarted;

    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private File                   mCascadeFile;
    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);
        accRateText = (TextView)this.findViewById(R.id.accel_rate);
        //magRateText = (TextView)this.findViewById(R.id.magn_rate);
        gyroRateText = (TextView)this.findViewById(R.id.gyro_rate);
        gpsStateText = (TextView)this.findViewById(R.id.gps_state);
        linearRateText = (TextView)this.findViewById(R.id.pressure_state);
        recodeStateText = (TextView)this.findViewById(R.id.recodestate);
        sensorStateText = (TextView)this.findViewById(R.id.sensorstate);
        startstopButton = (Button)this.findViewById(R.id.buttonStart);
        startstopButton.setOnClickListener(this);
        videotoggle = (CompoundButton)this.findViewById(R.id.videotoggle);
        videotoggle.setChecked(true);
        videotoggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                // 状態が変更された
                videoRecode = isChecked;
                //Toast.makeText(MainActivity.this, "isChecked : " + isChecked, Toast.LENGTH_SHORT).show();
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setMaxFrameSize(640, 480);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //GPSの起動
        locManager = (LocationManager)this.getSystemService(Context.LOCATION_SERVICE);
        Criteria criteria = new Criteria();
        criteria.setAccuracy(Criteria.ACCURACY_COARSE);
        criteria.setSpeedRequired(true);
        if(locManager != null){
            locManager.addNmeaListener(this); //NMEAも取得
            //locManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, GPS_INTERVAL , 0, this);
            locManager.requestLocationUpdates(GPS_INTERVAL, 0, criteria, this, null);

            //GPSがオンになっているか確認、表示
            if(locManager.isProviderEnabled(LocationManager.GPS_PROVIDER) == false){
                gpsStateText.setText(R.string.unavailable);
            }else{
                gpsStateText.setText(R.string.gps_unfix);
            }
        }

        //加速度／磁気センサの起動 + gryo
        sensorManager = (SensorManager)this.getSystemService(Context.SENSOR_SERVICE);

        List<Sensor> list = sensorManager.getSensorList(Sensor.TYPE_ACCELEROMETER);
        if(list.size()>0){
            accelerometer = list.get(0);
            sensorManager.registerListener(this, accelerometer, SENSOR_INTERVAL);
        }
        list = sensorManager.getSensorList(Sensor.TYPE_LINEAR_ACCELERATION);
        if(list.size()>0){
            linearaccelerometer = list.get(0);
            sensorManager.registerListener(this, linearaccelerometer, SENSOR_INTERVAL);
        }
//        list = sensorManager.getSensorList(Sensor.TYPE_MAGNETIC_FIELD);
//        if(list.size()>0){
//        	magnetometer = list.get(0);
//        	sensorManager.registerListener(this, magnetometer, SENSOR_INTERVAL);
//        }
        list = sensorManager.getSensorList(Sensor.TYPE_GYROSCOPE_UNCALIBRATED);
        if (list.size() > 0){
            gyroscope = list.get(0);
            sensorManager.registerListener(this,gyroscope,SENSOR_INTERVAL);
        }
//		list = sensorManager.getSensorList(Sensor.TYPE_PRESSURE);
//		if (list.size() > 0){
//			pressure = list.get(0);
//			sensorManager.registerListener(this,pressure,SENSOR_INTERVAL);
//		}

        //ストレージのパスを取得
        SDCardPath = Environment.getExternalStorageDirectory().getPath() + "/DrivingRecorder";
        File dir = new File(SDCardPath);
        if(!dir.exists()){ dir.mkdir(); }
        SDCardPath += "/";

        startstopButton.setVisibility(View.VISIBLE);
        logStarted = false;
        recordStarted = false;

        //センサキャリブレーションの指示
        Toast.makeText(this, R.string.please_calibrate, Toast.LENGTH_LONG ).show();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        //GPSとセンサを停止
        if(locManager != null){
            locManager.removeUpdates(this);
            locManager.removeNmeaListener(this);
        }
        if(sensorManager != null){
            sensorManager.unregisterListener(this);
        }

        if(recordStarted)
        {
            //Writerのクローズ
            if(gpsWriter!=null){
                gpsWriter.close();
                MediaScannerConnection.scanFile(this, new String[] { gpsFile.getAbsolutePath() }, null, null);
            }
            if(imglistWriter!=null){
                imglistWriter.close();
                MediaScannerConnection.scanFile(this, new String[] { imglistFile.getAbsolutePath() }, null, null);
            }
        }

        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (logStarted && videoRecode) {
            Imgproc.resize(mGray, mGray, new Size(mGray.size().width * 0.5, mGray.size().height * 0.5));

            Bitmap bmp = null;
            try {
                bmp = Bitmap.createBitmap(mGray.cols(), mGray.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mGray, bmp);
            } catch (CvException e) {
                Log.d(TAG, e.getMessage());
            }

            FileOutputStream out = null;
            long currentTimeMillis = System.currentTimeMillis();
            Date date = new Date(currentTimeMillis);
            SimpleDateFormat simpleDateFormat = new SimpleDateFormat("YYYYMMDD_HH_mm_ss_SSS");

            String timestamp = simpleDateFormat.format(date);
            String filename = timestamp + ".png";

            File sd = new File(recordfilePath);
            boolean success = true;
            if (!sd.exists()) {
                success = sd.mkdir();
            }
            if (success) {
                File dest = new File(sd, filename);

                try {
                    out = new FileOutputStream(dest);
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
                    imglistWriter.println(timestamp + COMMA + filename);
                    // PNG is a lossless format, the compression factor (100) is ignored

                } catch (Exception e) {
                    e.printStackTrace();
                    Log.d(TAG, e.getMessage());
                } finally {
                    try {
                        if (out != null) {
                            out.close();
                            //Log.d(TAG, "OK!!");
                        }
                    } catch (IOException e) {
                        Log.d(TAG, e.getMessage() + "Error");
                        e.printStackTrace();
                    }
                }
            }
        }

        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if(!logStarted){
            starttimestamp = event.timestamp;
            //磁気（or磁気センサ）の異常を検出
            if(event.sensor.getType()==Sensor.TYPE_MAGNETIC_FIELD){

                //加速度センサのサンプリングレートの異常を検出
            }else if(event.sensor.getType()==Sensor.TYPE_ACCELEROMETER ){
                if(accelRefTime == 0){
                    accelRefTime = event.timestamp;
                }else{
                    long elapse = event.timestamp - accelRefTime;
                    accelCount++;
                    if(elapse > 999999999/*[nsec]*/){

                        accRateText.setText(accelCount + "HZ");
                        if(accelCount < 20 || 60 < accelCount){ //22Hz以下もしくは42Hz以上の時
                            sensorStateText.setText(R.string.unavailable );
                        }else{
                            sensorStateText.setText(R.string.sensorstate_label);
                        }
                        accelCount = 0;
                        accelRefTime = event.timestamp;
                    }
                }
                // gyro scope data
            } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED){
                if(gyroRefTime == 0){
                    gyroRefTime = event.timestamp;
                }else{
                    long elapse = event.timestamp - gyroRefTime;
                    gyroCount++;
                    if(elapse > 999999999/*[nsec]*/){

                        gyroRateText.setText(gyroCount + "HZ");
                        if(gyroCount < 20 || 60 < gyroCount){ //22Hz以下もしくは42Hz以上の時
                            sensorStateText.setText(R.string.unavailable );
                        }else{
                            sensorStateText.setText(R.string.sensorstate_label);
                        }
                        gyroCount = 0;
                        gyroRefTime = event.timestamp;
                    }
                }

            } else if(event.sensor.getType()==Sensor.TYPE_LINEAR_ACCELERATION ) {
                if (linearRefTime == 0) {
                    linearRefTime = event.timestamp;
                } else {
                    long elapse = event.timestamp - linearRefTime;
                    linearCount++;
                    if (elapse > 999999999/*[nsec]*/) {

                        linearRateText.setText(linearCount + "HZ");
                        if (linearCount < 20 || 60 < linearCount) { //22Hz以下もしくは42Hz以上の時
                            sensorStateText.setText(R.string.unavailable);
                        } else {
                            sensorStateText.setText(R.string.sensorstate_label);
                        }
                        linearCount = 0;
                        linearRefTime = event.timestamp;
                    }
                }
            }

            //記録時（uptime_nanoにもtimestampを格納）
        }else{
//            if (!recordStarted){
//                startSavedata();
//            }
            //加速度
            if(event.sensor.getType()==Sensor.TYPE_ACCELEROMETER){
                uptime_nano = event.timestamp;

                //LINEAR_ACCELERATION
            }else if(event.sensor.getType()==Sensor.TYPE_LINEAR_ACCELERATION){

                if(recordStarted){
                    Log.i("ACCELERATION", event.values[0] + COMMA //X軸加速度 .. float[m/s^2]
                            + event.values[1] + COMMA //Y軸加速度 .. float[m/s^2]
                            + event.values[2] + COMMA //Z軸加速度 .. float[m/s^2]);
                    );
                }

                //磁気
            }else if(event.sensor.getType()==Sensor.TYPE_MAGNETIC_FIELD){


            }else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE_UNCALIBRATED){

                if(recordStarted){
                    Log.i("GYROSCOPE", (event.values[0] - event.values[3]) + COMMA
                            + (event.values[1] - event.values[4]) + COMMA
                            + (event.values[2] - event.values[5]) + COMMA);
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onNmeaReceived(long timestamp, String nmea) {
        String[] data = nmea.split(COMMA , -1);
        if(data[0].equals(NMEA_GPGSA) ){
            nmeaPdop = data[GSA_PDOP_COLUMN ];
            nmeaHdop = data[GSA_HDOP_COLUMN ];
            nmeaVdop = data[GSA_VDOP_COLUMN ].split("\\*")[0];
        }else if(data[0].equals(this.NMEA_GPGGA)){
            this.nmeaUTC = data[this.GGA_UTC_COLUMN];
        }
    }

    @Override
    public void onLocationChanged(Location loc) {
        //条件入力時
        if(!logStarted){
            //一定の精度が得られている場合のみ測定開始可能になる
            if(loc.hasAccuracy() && loc.getAccuracy() < START_ACCURACY_THRESH){
                gpsStateText.setText(R.string.gps_ok);
                if(isFixed == false){
                    startstopButton.setVisibility(View.VISIBLE);
                    isFixed = true;
                }
            }else{
                gpsStateText.setText(R.string.gps_unfix);
                if(isFixed == true){
                    startstopButton.setVisibility(View.INVISIBLE);
                    isFixed = false;
                }
            }
            if(loc.hasAccuracy()){
                gpsStateText.append(Integer.toString((int)loc.getAccuracy()));
            }


            //記録時（uptime_nanoが0の時は無効）DOPも並べて記録する
        }else{
            if(uptime_nano!=0){
                gpsWriter.println(
                        (uptime_nano - starttimestamp) + COMMA  //UPTIMENANO .. long[nsec]
                                + loc.getTime() + COMMA  //GPSTIME .. long[msec] ※UNIX時間
                                + (loc.hasSpeed() ? loc.getSpeed() : -1)+ COMMA //SPEED .. float[m/s] ※データ無しの場合は-1
                                + (float)loc.getLatitude() + COMMA //LAT .. double->float
                                + (float)loc.getLongitude() + COMMA //LON .. double->float
                                + (float)loc.getAltitude() + COMMA //ALT .. double->float
                                + loc.getBearing() + COMMA //BEARING .. float[deg]
                                + loc.getAccuracy() + COMMA //ACCURACY .. float[m]
                                + nmeaPdop + COMMA //PDOP[]
                                + nmeaHdop + COMMA //HDOP[]
                                + nmeaVdop + COMMA //VDOP[]
                                + this.nmeaUTC
                );
                //modeClassifier.onGPSUpdate((loc.hasSpeed() ? loc.getSpeed() : -1));
            }
        }
    }

    @Override
    public void onStatusChanged(String provider, int status, Bundle extras) {

    }

    @Override
    public void onProviderEnabled(String provider) {
        gpsStateText.setText(R.string.gps_unfix);
        isFixed = false;
        locManager.addNmeaListener(this);
        locManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, GPS_INTERVAL , 0, this);
    }

    @Override
    public void onProviderDisabled(String provider) {
        gpsStateText.setText(R.string.unavailable);
        isFixed = false;
    }

    @Override
    public void onClick(View v) {
        if(v==startstopButton){

            //測定開始時
            if(!logStarted){

                recodeStateText.setText(R.string.recoding);
                recodeStateText.setTextColor(Color.RED);

                startstopButton.setText(R.string.finish_logging);
                startstopButton.setTextColor(Color.BLUE);

                videotoggle.setVisibility(View.INVISIBLE);

                //測定開始時刻を取得
                long startUnixTime = System.currentTimeMillis();
                Date date = new Date(startUnixTime);
                SimpleDateFormat simpleDateFormat = new SimpleDateFormat("YYYYMMDD_HH_mm_ss_SSS");
                startTime = simpleDateFormat.format(date);
                //記録開始
                startSavedata();
                logStarted = true;

                //測定終了時
            }else{
                logStarted = false;
                if(recordStarted){
                    stopSavedata();
                }

                //「記録しました」Toast
                Toast.makeText(this, R.string.saved, Toast.LENGTH_SHORT).show();

                //アプリの終了
                this.finish();
            }
        }
    }
//    public String getCurrentYYYYMMDDhhmmss(){
//        Calendar cal = Calendar.getInstance();
//        int year = cal.get(Calendar.YEAR);
//        int month = cal.get(Calendar.MONTH) + 1; //MONTHは0-11で返されるため
//        int day = cal.get(Calendar.DAY_OF_MONTH);
//        int hour = cal.get(Calendar.HOUR_OF_DAY);
//        int minute = cal.get(Calendar.MINUTE);
//        int second = cal.get(Calendar.SECOND);
//        return String.format("%04d%02d%02d%02d%02d%02d",year,month,day,hour,minute,second);
//    }

    private void createLocalDirectory(){
        this.recordfilePath = this.SDCardPath + startTime;
        File dir = new File(recordfilePath);
        if(!dir.exists()){ dir.mkdir(); }
        recordfilePath += "/";
    }

    public boolean startSavedata(){
        createLocalDirectory();
        //現在時刻と選択された運動モードを取得

        //記録用のファイルを用意し、1行目を記入
        try{
            gpsFile = new File(recordfilePath  + "GPS" + startTime + ".csv");
            imglistFile = new File(recordfilePath + "Imglist" + startTime + ".csv");

            gpsWriter = new PrintWriter(new BufferedWriter(new FileWriter(gpsFile)));
            imglistWriter = new PrintWriter(new BufferedWriter(new FileWriter(imglistFile)));

            gpsWriter.println("UPTIMENANO,GPSTIME,SPEED,LAT,LON,ALT,BEARING,ACCURACY,PDOP,HDOP,VDOP,GPSUTC");
            imglistWriter.println("TimeStamp, ImgName");

            recordStarted = true;

        } catch (IOException e){
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public boolean stopSavedata(){
        recordStarted = false;
        //Writerのクローズ
        if(gpsWriter!=null){
            gpsWriter.close();
            MediaScannerConnection.scanFile(this, new String[] { gpsFile.getAbsolutePath() }, null, null);
        }
        if(imglistWriter!=null){
            imglistWriter.close();
            MediaScannerConnection.scanFile(this, new String[] { imglistFile.getAbsolutePath() }, null, null);
        }
        return true;
    }
}
