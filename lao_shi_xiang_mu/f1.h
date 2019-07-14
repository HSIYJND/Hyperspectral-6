#include "stdafx.h"
#include "Eigen\Core"
#include <iostream>
#include <fstream>
#include <set>
#include <vector>
#include<sstream>
using namespace Eigen;
using namespace std;

double cosine_similarity(VectorXd v1, VectorXd v2, set<int> t, int cur_index)
{
	// 判断某两个光谱曲线除了t中波段，由其余波段计算出的余弦相似度
	int l = v1.size();
	double res = 0;
	double m1 = 0;
	double m2 = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		if ((t.count(i) >= 1) | (i == cur_index))
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

void testReadTxt()
{
	string f;
	f = "C:/Users/82761/Downloads/lao_shi_xiang_mu/lao_shi_xiang_mu/res9.txt";
	readTxt(f);

	cout << 123 << endl;
}

set<int> selectBest(int k, VectorXd b, VectorXd a)
{
	set<int> selected;
	selected.insert(-1);
	while (selected.size() < 3)
	{
		int selected_index = -1; // 选最小的
		float min_score = -1.0;
		for (int i = 0; i < 5; i++)
		{
			float cur_score = 0.0;
			if (selected.count(i) > 0)
			{
				continue;
			}
			for (int j = 0; j < 10; j++)
			{
				cur_score += cosine_similarity(a[i][:], b, selected, i);

			}
			if ((cur_score < min_score) | (selected_index == -1))
			{
				selected_index = i;
				min_score = cur_score;
			}
		}
		selected.insert(selected_index);
	}
	return selected;
}
