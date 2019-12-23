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
		//���캯�������ݲ���
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
			//�������û����д����û�Ϊkey�����û�ÿ����һ����վ���� ���ڼ������+�ڼ�����վ+���ʴ������ĸ�ʽд��reduce��
			String str[] = value.toString().split("\t", 4);	 //��ԭʼ���ݽ��зִ�
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
			//�û�Ϊkeyֵ���û�������Ƶ��վ�ļ�¼����������Ϊvalueֵд���ļ�
			String tt = new String();
			for(Text t : values)
			{
				tt += t.toString();		//��ÿһ���û���Ϊһ����¼
			}
			tt = tt.substring(0, tt.length()-1);	//ȥ��ÿ���ַ�����������ð��
			context.write(key, new Text(tt));
		}
	}
	public void preProcess1() 
		throws Exception
	{
		Job job = new Job(conf, "preProcess1");
		job.setJarByClass(Tianyi.class);
		//����MapReduce��Mapper���Reducer��
		job.setMapperClass(Map1.class);
		job.setReducerClass(Reduce1.class);
		//��������ļ�ֵ����������
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		//�����������·��
		FileInputFormat.addInputPath(job, path1);
		FileOutputFormat.setOutputPath(job, path2);
		
		//job.setNumReduceTasks(5); ���������5��Map�Ա�ӿ��ٶȡ�
		//�ύ��ҵ
		job.waitForCompletion(true);
	}
}
