package com.fangding;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Preprocess {
	private Path path1;
	private Path path2;
	private Configuration conf;
	public Preprocess(Configuration c,Path p1,Path p2)
	{
		//构造函数，传递参数
		conf = c;
		path1 = p1;
		path2 = p2;
	}
	public static class Map1 extends Mapper<Object, Text, Text, Text>
	{
		private Text user_name = new Text();
		private Text user_data = new Text();
		int x,y; 
		public void map(Object key, Text value, Context context) 
			throws IOException, InterruptedException
		{
			//对所有用户进行处理，用户为key，，用户每访问一次网站就以 “第几天访问+第几个网站+访问次数”的格式写到reduce上
			String str[] = value.toString().split("\t", 4);	 //对原始数据进行分词
			user_name.set(str[0]);
			x = str[1].charAt(1) - '1';
			y = str[1].charAt(3) - '0';
			x = x*7+y;
			user_data.set(""+x+"+"+str[2].substring(1,str[2].length())+"+"+str[3]+":");
			context.write(user_name, user_data);
		}
	}
	public static class Reduce1 extends Reducer<Text, Text, Text, Text>
	{		
		public void reduce(Text key, Iterable<Text> values, Context context) 
			throws IOException, InterruptedException
		{
			//用户为key值，用户访问视频网站的记录连接起来作为value值写入文件
			String tt = new String();
			for(Text t : values)
			{
				tt += t.toString();		//是每一个用户成为一条记录
			}
			tt = tt.substring(0, tt.length()-1);	//去除每个字符串后面多余的冒号
			context.write(key, new Text(tt));
		}
	}
	public void preProcess1() 
		throws Exception
	{
		Job job = new Job(conf, "preProcess1");
		job.setJarByClass(Tianyi.class);
		//设置MapReduce的Mapper类和Reducer类
		job.setMapperClass(Map1.class);
		job.setReducerClass(Reduce1.class);
		//设置输出的键值对数据类型
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		//设置输入输出路径
		FileInputFormat.addInputPath(job, path1);
		FileOutputFormat.setOutputPath(job, path2);
		
		//job.setNumReduceTasks(5); 这个是设置5个Map以便加快速度。
		//提交作业
		job.waitForCompletion(true);
	}
}
