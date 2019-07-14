#include "stdafx.h"
#include "Eigen\Core"
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include<sstream>
using namespace Eigen;
using namespace std;

double cosine_similarity(VectorXd v1, VectorXd v2, set<int> t)
{
	// 判断某两个光谱曲线除了t中波段，由其余波段计算出的余弦相似度
	int l = v1.size();
	double res = 0;
	double m1 = 0;
	double m2 = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		if (t.count(i) == 1)
		{
			continue;
		}
		res = res + v1[i] * v2[i];
		m1 = m1 + v1[i] * v1[i];
		m2 = m2 + v2[i] * v2[i];
	}

	res = res / sqrt(m1) / sqrt(m2);
	return res;

}


void readTxt(string file)
{
	ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	string s;
	while (getline(infile, s))
	{
		cout << s << endl;
	}
	infile.close();             //关闭文件输入流 
}


vector<double> str2double(string line)
{
	vector<double> v;
	string cur;
	double res;
	
	stringstream input(line);
	while (input >> cur)
	{
		res = atoi(cur.c_str());
		v.push_back(res);
	}
	return v;
}
