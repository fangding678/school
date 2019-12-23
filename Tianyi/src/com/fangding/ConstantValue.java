package com.fangding;

import org.apache.hadoop.fs.Path;
//把所有的常量写在一个类中，方便程序后边的维护。当我们需要改变常量的时候就显得十分方便。
public class ConstantValue {
	public static final String pStr = new String("hdfs://192.168.248.140:9000/user/TianYi/");
	//之前的老数据路径。分别是输入文件路径、预处理后文件路径、预测结果路径
	public static final Path pathRawOldInput = new Path("hdfs://192.168.248.140:9000/user/TianYi/tianyiOldData");
	public static final Path pathOldUser = new Path("hdfs://192.168.248.140:9000/user/TianYi/classifyOldData");
	public static final Path pathOldResult = new Path("hdfs://192.168.248.140:9000/user/TianYi/resultOldData");
	//新数据路径。分别是输入文件路径、预处理后文件路径、预测结果路径
	public static final Path pathRawNewInput = new Path("hdfs://192.168.248.140:9000/user/TianYi/tianyiNewData");
	public static final Path pathNewUser = new Path("hdfs://192.168.248.140:9000/user/TianYi/classifyNewData");
	public static final Path pathNewResult = new Path("hdfs://192.168.248.140:9000/user/TianYi/resultNewData");
}
