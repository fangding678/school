package com.fangding;

import org.apache.hadoop.fs.Path;
//�����еĳ���д��һ�����У���������ߵ�ά������������Ҫ�ı䳣����ʱ����Ե�ʮ�ַ��㡣
public class ConstantValue {
	public static final String pStr = new String("hdfs://192.168.248.140:9000/user/TianYi/");
	//֮ǰ��������·�����ֱ��������ļ�·����Ԥ������ļ�·����Ԥ����·��
	public static final Path pathRawOldInput = new Path("hdfs://192.168.248.140:9000/user/TianYi/tianyiOldData");
	public static final Path pathOldUser = new Path("hdfs://192.168.248.140:9000/user/TianYi/classifyOldData");
	public static final Path pathOldResult = new Path("hdfs://192.168.248.140:9000/user/TianYi/resultOldData");
	//������·�����ֱ��������ļ�·����Ԥ������ļ�·����Ԥ����·��
	public static final Path pathRawNewInput = new Path("hdfs://192.168.248.140:9000/user/TianYi/tianyiNewData");
	public static final Path pathNewUser = new Path("hdfs://192.168.248.140:9000/user/TianYi/classifyNewData");
	public static final Path pathNewResult = new Path("hdfs://192.168.248.140:9000/user/TianYi/resultNewData");
}
