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
		conf.set("mapred.jar", "D://Java//Master2//Tianyi.jar");	//在eclipse下用ant打包使用
		conf.set("mapred.job.tracker", "192.168.248.140:9001");
		//老数据路径 : cv.pathRawOldInput, cv.pathOldUser, cv.pathOldResult
//		Path pathInProcess = cv.pathRawOldInput;
//		Path pathInPredict = cv.pathOldUser;
//		Path pathOutPredict = cv.pathOldResult;
		//新数据路径 : cv.pathRawNewInput, cv.pathNewUser, cv.pathNewResult
		Path pathInProcess = cv.pathRawNewInput;
		Path pathInPredict = cv.pathNewUser;
		Path pathOutPredict = cv.pathNewResult;
		//预处理数据
//		Preprocess preprocess = new Preprocess(conf, pathInProcess, pathInPredict);
//		preprocess.preProcess1();
		
		//对数据进行预测
		Predict predict = new Predict(conf, pathInPredict, pathOutPredict);
		predict.predict1();
	}

}
