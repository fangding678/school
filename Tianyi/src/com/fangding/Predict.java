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
		//构造函数，传递参数
		conf = c;
		pInput = p1;
		pOutput = p2;
	}

	public static class Map1 extends Mapper<Object, Text, Text, Text> 
	{
		String keyStr = new String();
		String valueStr = new String();
		double[] w1 = new double[8];  	//第七周的权值
		double[] w2 = new double[8];	//七周中星期的权值
		
		@Override
		protected void setup(Mapper<Object, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {
			super.setup(context);
			//算法的参数在这里调节。不同的参数值会得到不同的结果。
			//很显然，当模型固定的时候，参数的选择将决定比赛的胜负
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
			int arr[][] = new int[11][50];	//二维数组保存访问视频次数的情况
			int ar[][] = new int[11][50];	//二维数组保存是否访问视频的情况
			int re[][] = new int[11][8];	//保存预测结果数组
			String str[] = ss[1].split(":");
			for(String s : str)
			{
				//这里有一个关键的地方值得注意，使用split函数进行分词的时候"+"是转义字符。
				//使用StringTokerier粉刺就不会出现这种情况
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
			for(int i=1; i<=10; ++i)	//十个视频网站
			{
				for(int j=1; j<=7; ++j)		//第八周的七天
				{
					weight = 0.0;
					wt = 0.0;
					//对每一周的这一天和第七周的七天分别求权值
					for(int k=1; k<=7; ++k)	
					{
						weight += ar[i][42+k]*w1[k] + ar[i][(k-1)*7+j]*w2[k];
						wt += arr[i][42+k]*w1[k] + arr[i][(k-1)*7+j]*w2[k];
					}
					re[i][j] = 0;
					if(weight > 1.0) 	//访问网站概率大于P，我们预测用户会访问网站
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
				//按照比赛要求格式将结果写入文件
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
		//没有Reducer程序会在Map过程写入文件，所以设置输出类
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		//这个程序的map过程就已经得到了最终结果。所以reducer过程就不用了。
//		job.setOutputKeyClass(Text.class);
//		job.setOutputValueClass(Text.class);
		//设置输入输出路径
		FileInputFormat.setInputPaths(job, pInput);
		FileOutputFormat.setOutputPath(job, pOutput);
		//提交job
		job.waitForCompletion(true);
	}
}