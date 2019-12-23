package com.fangding;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

public class Tianyi {
/*	Master2
	hdfs://192.168.248.140:9000/user/TianYi/train
	hdfs://192.168.248.140:9000/user/TianYi/test
	hdfs://192.168.248.140:9000/user/TianYi/tianyiNewData

*/
	@SuppressWarnings("static-access")
	public static void main(String[] args) throws Exception 
	{
		// TODO Auto-generated method stub
		Configuration conf = new Configuration();
		ConstantValue cv = new ConstantValue();
		conf.set("mapred.jar", "D://Java//Master2//Tianyi.jar");	//��eclipse����ant���ʹ��
		conf.set("mapred.job.tracker", "192.168.248.140:9001");
		//������·�� : cv.pathRawOldInput, cv.pathOldUser, cv.pathOldResult
//		Path pathInProcess = cv.pathRawOldInput;
//		Path pathInPredict = cv.pathOldUser;
//		Path pathOutPredict = cv.pathOldResult;
		//������·�� : cv.pathRawNewInput, cv.pathNewUser, cv.pathNewResult
		Path pathInProcess = cv.pathRawNewInput;
		Path pathInPredict = cv.pathNewUser;
		Path pathOutPredict = cv.pathNewResult;
		//Ԥ��������
//		Preprocess preprocess = new Preprocess(conf, pathInProcess, pathInPredict);
//		preprocess.preProcess1();
		
		//�����ݽ���Ԥ��
		Predict predict = new Predict(conf, pathInPredict, pathOutPredict);
		predict.predict1();
	}

}
