// arraylearn.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Eigen\Core"
#include <iostream>
//#include "f1.h"
#include <math.h>
#include <fstream>
#include <set>
#include <mat.h>
#include <vector>
using namespace Eigen;
using namespace std;

void main2()
{
	string s;
	s = "123";
	cout << s << endl;
	int a;
	a = atoi(s.c_str());
	cout << a + 1 << endl;
	string s2;
	s2 = "123   456       789";

	vector<double> v1;
	string res;
	double cur;

	stringstream input(s2);
	while (input >> res)
	{
		cur = atoi(res.c_str());
		v1.push_back(cur);
	}
	for (int i = 0; i < v1.size(); i++)
	{
		cout << v1[i]+0.1 << endl;
	}
}

