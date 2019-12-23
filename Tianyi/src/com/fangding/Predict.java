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

public class Predict {
	Configuration conf;
	private Path pInput;
	private Path pOutput;

	public Predict(Configuration c, Path p1, Path p2) 
	{
		//���캯�������ݲ���
		conf = c;
		pInput = p1;
		pOutput = p2;
	}

	public static class Map1 extends Mapper<Object, Text, Text, Text> 
	{
		String keyStr = new String();
		String valueStr = new String();
		double[] w1 = new double[8];  	//�����ܵ�Ȩֵ
		double[] w2 = new double[8];	//���������ڵ�Ȩֵ
		
		@Override
		protected void setup(Mapper<Object, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			//�㷨�Ĳ�����������ڡ���ͬ�Ĳ���ֵ��õ���ͬ�Ľ����
			//����Ȼ����ģ�͹̶���ʱ�򣬲�����ѡ�񽫾���������ʤ��
			double ss1 = 0.0;
			double ss2 = 0.0;
			for(int ii = 1;ii<=7;++ii)
			{
				ss1 += Math.pow(ii, 1.2);
				ss2 += Math.pow(ii, 1.1);
			}
			for(int ii = 1;ii<=7;++ii)
			{
				w1[ii] = Math.pow(ii, 1.3) / ss1;
				w2[ii] = Math.pow(ii, 1.2) / ss2;
			}
		}
		
		@Override
		public void map(Object key, Text values, Context context) 
			throws IOException, InterruptedException
		{
			String ss[] = values.toString().split("\t");
			keyStr = ss[0];
			valueStr = "";
			int day;
			int vedio;
			int times;
			int arr[][] = new int[11][50];	//��ά���鱣�������Ƶ���������
			int ar[][] = new int[11][50];	//��ά���鱣���Ƿ������Ƶ�����
			int re[][] = new int[11][8];	//����Ԥ��������
			String str[] = ss[1].split(":");
			for(String s : str)
			{
				//������һ���ؼ��ĵط�ֵ��ע�⣬ʹ��split�������зִʵ�ʱ��"+"��ת���ַ���
				//ʹ��StringTokerier�۴̾Ͳ�������������
				String st[] = s.split("\\+");
				day = Integer.parseInt(st[0]);
				vedio = Integer.parseInt(st[1]);
				times = Integer.parseInt(st[2]);
				arr[vedio][day] = times;
				ar[vedio][day] = 1;
			}
			double weight = 0.0;
			double wt = 0.0;
			boolean flag = false;
			for(int i=1; i<=10; ++i)	//ʮ����Ƶ��վ
			{
				for(int j=1; j<=7; ++j)		//�ڰ��ܵ�����
				{
					weight = 0.0;
					wt = 0.0;
					//��ÿһ�ܵ���һ��͵����ܵ�����ֱ���Ȩֵ
					for(int k=1; k<=7; ++k)	
					{
						weight += ar[i][42+k]*w1[k] + ar[i][(k-1)*7+j]*w2[k];
						wt += arr[i][42+k]*w1[k] + arr[i][(k-1)*7+j]*w2[k];
					}
					re[i][j] = 0;
					if(weight > 1.0) 	//������վ���ʴ���P������Ԥ���û��������վ
					{
						re[i][j] = (int) Math.rint(wt);
						flag = true;
					}
				}
			}
			valueStr = "";
			if(flag)
			{
				for(int i=1; i<=7; ++i)
				{
					for(int j=1; j<=10; ++j)
					{
						valueStr += String.valueOf(re[j][i]) + ",";
					}
				}
				valueStr = valueStr.substring(0,valueStr.length()-1);
				//���ձ���Ҫ���ʽ�����д���ļ�
				context.write(new Text(keyStr), new Text(valueStr));
			}			
		}
	}

	public static class Reduce1 extends Reducer<Text, Text, Text, Text> 
	{
		public void reduce(Text key, Iterable<Text> values, Context context) 
		{
			;
		}
	}

	public void predict1() throws Exception 
	{
		Job job = new Job(conf, "predict1");
		job.setJarByClass(Tianyi.class);
		job.setMapperClass(Map1.class);
		//job.setReducerClass(Reduce1.class);
		//û��Reducer�������Map����д���ļ����������������
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		//��������map���̾��Ѿ��õ������ս��������reducer���̾Ͳ����ˡ�
//		job.setOutputKeyClass(Text.class);
//		job.setOutputValueClass(Text.class);
		//�����������·��
		FileInputFormat.setInputPaths(job, pInput);
		FileOutputFormat.setOutputPath(job, pOutput);
		//�ύjob
		job.waitForCompletion(true);
	}
}