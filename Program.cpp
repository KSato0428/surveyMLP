//========================================================================
// This conversion was produced by the Free Edition of
// C# to C++ Converter courtesy of Tangible Software Solutions.
// Order the Premium Edition at https://www.tangiblesoftwaresolutions.com
//========================================================================

#include "Program.h"


namespace MLP
{

	void MainProgram::Main(std::vector<std::wstring> &args)
	{
		auto sw = new System::Diagnostics::Stopwatch();
		Random *rnd = new Random(2);
		std::function<double()> genRnd = [&] ()
		{
			return rnd->NextDouble() - 0.5;
		};
		std::function<bool(double, double, double)> withIn = [] (t, x, y)
		{
			return x * x + y * y < t * t;
		};

		MLP *nn = new MLP(2, std::vector<int> {2, 5, 2});

		int sampleSize = 4000;
		int MaxIt = 10000;

		double r = 30, pr = 10;
		std::vector<std::vector<double>> sample(sampleSize);
		for (int i = 0; i < sampleSize; i++)
		{
			sample[i] = std::vector<double>(2);
		}
		std::vector<std::vector<double>> sampleLabel(sampleSize);
		for (int i = 0; i < sampleSize; i++)
		{
			sampleLabel[i] = std::vector<double>(2);
		}
		for (int i = 0; i < sampleSize; i++)
		{
			double x = r * genRnd(), y = r * genRnd();
			sample[i][0] = x;
			sample[i][1] = y;
			sampleLabel[i][0] = withIn(pr, x, y) ? 0 : 1;
			sampleLabel[i][1] = (1 + sampleLabel[i][0]) % 2;
		}
		sw->Start();
		//training
		for (int p = 0; p < MaxIt; p++)
		{
			for (int i = 0; i < sampleSize; i++)
			{
				nn->train(sample[i], sampleLabel[i]);
			}

			int tcnt = 0;
			for (int k = 0; k < sampleSize; k++)
			{
				std::vector<double> nyn = nn->Predict(sample[k]);
				double c1 = nyn[0] * nyn[0] + (nyn[1] - 1) * (nyn[1] - 1);
				double c2 = (nyn[0] - 1) * (nyn[0] - 1) + nyn[1] * nyn[1];
				if (sampleLabel[k][0] == 0 && c1 < c2)
				{
					tcnt++;
				}
				if (sampleLabel[k][0] == 1 && c2 < c1)
				{
					tcnt++;
				}
			}
			if (p % 100 == 0)
			{
				std::wcout << std::to_wstring(tcnt) << L"/" << std::to_wstring(sampleSize) << std::endl;
			}

		}
		sw->Stop();
		std::wcout << std::wstring::Format(L"{0} s", sw->ElapsedMilliseconds / 1000.0) << std::endl;

		int cnt = 0;
		int AL = 1000;
		for (int i = 0; i < AL; i++)
		{
			double x = r * genRnd(), y = r * genRnd();
			std::vector<double> nyn = nn->Predict(std::vector<double> {x, y});
			double c1 = nyn[0] * nyn[0] + (nyn[1] - 1) * (nyn[1] - 1);
			double c2 = (nyn[0] - 1) * (nyn[0] - 1) + nyn[1] * nyn[1];
			if (withIn(pr, x, y))
			{
				if (c1 < c2)
				{
					cnt++;
				}
			}
			else
			{
				if (c2 < c1)
				{
					cnt++;
				}
			}
		}
		std::wcout << cnt << std::endl;
		std::wcout << AL - cnt << std::endl;

		nn->Check({4, 4}, std::vector<double> {0, 1});
		nn->Check({-4, 4}, std::vector<double> {0, 1});
		nn->Check({14, -4}, std::vector<double> {1, 0});
	}

	MLP::MLP(int D, std::vector<int> &M)
	{
		rnd = new Random(1);
		this->D = D;
		this->M = std::vector<int>(D + 1);
		for (int i = 0; i < D + 1; i++)
		{
			this->M[i] = M[i];
		}
		this->a = std::vector<std::vector<double>>(D);
		for (int i = 0; i < D; i++)
		{
			this->a[i] = std::vector<double>(this->M[i + 1]);
		}
		this->o = std::vector<std::vector<double>>(D + 1);
		for (int i = 0; i < D + 1; i++)
		{
			this->o[i] = std::vector<double>(this->M[i]);
		}
		this->delta = std::vector<std::vector<double>>(D);
		for (int i = 0; i < D; i++)
		{
			this->delta[i] = std::vector<double>(M[i + 1]);
		}
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: this.dw = new double[D][,];
		this->dw = RectangularVectors::ReturnRectangularDoubleVector(D, ,);
		for (int i = 0; i < D; i++)
		{
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: this.dw[i] = new double[M[i], M[i + 1]];
			this->dw[i] = RectangularVectors::ReturnRectangularDoubleVector(M[i], M[i + 1]);
		}
		this->dth = std::vector<std::vector<double>>(D);
		for (int i = 0; i < D; i++)
		{
			this->dth[i] = std::vector<double>(M[i + 1]);
		}

		//Initialization of parameters
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: this.w = new double[D][,];
		this->w = RectangularVectors::ReturnRectangularDoubleVector(D, ,);
		for (int i = 0; i < D; i++)
		{
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: this.w[i] = new double[M[i], M[i + 1]];
			this->w[i] = RectangularVectors::ReturnRectangularDoubleVector(M[i], M[i + 1]);
			for (int j = 0; j < M[i]; j++)
			{
				for (int k = 0; k < M[i + 1]; k++)
				{
					this->w[i][j][k] = rnd->NextDouble() - 0.5;
				}
			}
		}
		this->theta = std::vector<std::vector<double>>(D);
		for (int i = 0; i < D; i++)
		{
			this->theta[i] = std::vector<double>(M[i + 1]);
			for (int j = 0; j < M[i + 1]; j++)
			{
				this->theta[i][j] = rnd->NextDouble() - 0.5;
			}
		}
	}

	double MLP::sig(double x)
	{
		return 1.0 / (1.0 + std::exp(-x));
	}

	double MLP::sign(double x)
	{
		return x > 0 ? 1.0 : -1.0;
	}

	double MLP::sig_grad(double x)
	{
		return sig(x) * (1.0 - sig(x));
	}

	void MLP::predict(std::vector<double> &x)
	{
		for (int j = 0; j < M[0]; j++)
		{
			o[0][j] = x[j];
		}
		for (int i = 0; i < D; i++)
		{
			for (int kn = 0; kn < M[i + 1]; kn++)
			{
				a[i][kn] = theta[i][kn];
				for (int kb = 0; kb < M[i]; kb++)
				{
					a[i][kn] += o[i][kb] * w[i][kb][kn];
				}
				o[i + 1][kn] = sig(a[i][kn]);
			}
		}
	}

	std::vector<double> MLP::Predict(std::vector<double> &x)
	{
		predict(x);
		std::vector<double> ans(M[D]);
		for (int i = 0; i < M[D]; i++)
		{
			ans[i] = o[D][i];
		}
		return ans;
	}

	void MLP::gradient(std::vector<double> &x, std::vector<double> &y)
	{
		predict(x);
		for (int k = 0; k < M[D]; k++)
		{
			delta[D - 1][k] = (o[D][k] - y[k]) * o[D][k] * (1.0 - o[D][k]);
		}
		for (int i = D - 2; i > -1; i--)
		{
			for (int k = 0; k < M[i + 1]; k++)
			{
				delta[i][k] = 0.0;
				for (int l = 0; l < M[i + 2]; l++)
				{
					delta[i][k] += w[i + 1][k][l] * delta[i + 1][l];
				}
				delta[i][k] = delta[i][k] * o[i + 1][k] * (1.0 - o[i + 1][k]);
			}
		}
		for (int i = 0; i < D; i++)
		{
			for (int k = 0; k < M[i + 1]; k++)
			{
				dth[i][k] = delta[i][k];
				for (int j = 0; j < M[i]; j++)
				{
					dw[i][j][k] = delta[i][k] * o[i][j];
				}
			}
		}
	}

	void MLP::Check(std::vector<double> &x, std::vector<double> &y)
	{
		gradient(x, y);
		double epsilon = 0.00001;
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: double[][,] ddw = new double[D][,];
		std::vector<std::vector<std::vector<double>>> ddw = RectangularVectors::ReturnRectangularDoubleVector(D, ,);
		for (int i = 0; i < D; i++)
		{
//C# TO C++ CONVERTER NOTE: The following call to the 'RectangularVectors' helper class reproduces the rectangular array initialization that is automatic in C#:
//ORIGINAL LINE: ddw[i] = new double[M[i], M[i + 1]];
			ddw[i] = RectangularVectors::ReturnRectangularDoubleVector(M[i], M[i + 1]);
		}

		std::vector<double> bw, ew;
		for (int i = 0; i < D; i++)
		{
			for (int j = 0; j < M[i]; j++)
			{
				for (int k = 0; k < M[i + 1]; k++)
				{
					double defw = w[i][j][k];
					w[i][j][k] = defw + epsilon;
					bw = Predict(x);
					w[i][j][k] = defw - epsilon;
					ew = Predict(x);
					w[i][j][k] = defw;

					double error1 = 0, error2 = 0;
					for (int s = 0; s < M[D]; s++)
					{
						error1 += 0.5 * (bw[s] - y[s]) * (bw[s] - y[s]);
						error2 += 0.5 * (ew[s] - y[s]) * (ew[s] - y[s]);
					}
					ddw[i][j][k] = (error1 - error2) / (2 * epsilon);
				}
			}
		}

		double sm1 = 0, sm2 = 0, sm3 = 0;
		for (int i = 0; i < D; i++)
		{
			for (int j = 0; j < M[i]; j++)
			{
				for (int k = 0; k < M[i + 1]; k++)
				{
					sm1 += (ddw[i][j][k] - dw[i][j][k]) * (ddw[i][j][k] - dw[i][j][k]);
					sm2 += ddw[i][j][k] * ddw[i][j][k];
					sm3 += dw[i][j][k] * dw[i][j][k];
				}
			}
		}
		double an = std::sqrt(sm1) / (std::sqrt(sm2) + std::sqrt(sm3) + epsilon * epsilon);
		std::wcout << L"Accuracy:" << std::to_wstring(an) << std::endl;
	}

	void MLP::train(std::vector<double> &x, std::vector<double> &y)
	{
		gradient(x, y);
		for (int i = 0; i < D; i++)
		{
			for (int k = 0; k < M[i + 1]; k++)
			{
				theta[i][k] -= eta * (dth[i][k] + l1 * sign(theta[i][k]) + l2 * theta[i][k]);
				for (int j = 0; j < M[i]; j++)
				{
					w[i][j][k] -= eta * (dw[i][j][k] + l1 * sign(w[i][j][k]) + l2 * w[i][j][k]);
				}
			}
		}
	}
}
