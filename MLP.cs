using System;

namespace MLP {
  public class MainProgram {
    static void Main(string[] args){
      var sw = new System.Diagnostics.Stopwatch();
      Random rnd = new Random(2);
      Func<double> genRnd = () => rnd.NextDouble()-0.5;
      Func<double,double,double, bool> withIn = (t, x, y) => x*x+y*y<t*t;

      MLP nn = new MLP(2, new int[]{2,5,2});

      int sampleSize = 4000;
      int MaxIt = 2000;

      double r = 30, pr = 10;
      double[][] sample = new double[sampleSize][];
      for(int i=0;i<sampleSize;i++) sample[i] = new double[2];
      double[][] sampleLabel = new double[sampleSize][];
      for(int i=0;i<sampleSize;i++) sampleLabel[i] = new double[2];
      for(int i=0;i<sampleSize;i++){
        double x = r * genRnd(), y = r * genRnd();
        sample[i][0] = x; sample[i][1] = y;
        sampleLabel[i][0] = withIn(pr, x, y) ? 0 : 1;
        sampleLabel[i][1] = (1+sampleLabel[i][0])%2;
      }
      sw.Start();
      //training
      for(int p=0;p<MaxIt;p++){
        for(int i=0;i<sampleSize;i++) nn.train(sample[i], sampleLabel[i]);

        int tcnt = 0;
        for(int k=0;k<sampleSize;k++){
          double[] nyn = nn.Predict(sample[k]);
          double c1 = nyn[0]*nyn[0]+(nyn[1]-1)*(nyn[1]-1);
          double c2 = (nyn[0]-1)*(nyn[0]-1)+nyn[1]*nyn[1];
          if(sampleLabel[k][0]==0 && c1 < c2) tcnt++;
          if(sampleLabel[k][0]==1 && c2 < c1) tcnt++;
        }
        if(p%100==0) Console.WriteLine(tcnt.ToString() + "/"+sampleSize.ToString());

      }
      sw.Stop();
      Console.WriteLine($"{sw.ElapsedMilliseconds/1000.0} s");

      int cnt = 0;
      int AL = 1000;
      for(int i=0;i<AL;i++){
        double x = r*genRnd(), y = r*genRnd();
        double[] nyn = nn.Predict(new double[]{x,y});
        double c1 = nyn[0]*nyn[0]+(nyn[1]-1)*(nyn[1]-1);
        double c2 = (nyn[0]-1)*(nyn[0]-1)+nyn[1]*nyn[1];
        if(withIn(pr,x,y)){
          if(c1<c2) cnt++;
        } else {
          if(c2<c1) cnt++;
        }
      }
      Console.WriteLine(cnt);
      Console.WriteLine(AL-cnt);

      nn.Check(new double[]{4,4}, new double[]{0,1});
      nn.Check(new double[]{-4,4}, new double[]{0,1});
      nn.Check(new double[]{14,-4}, new double[]{1,0});
    }
  }
  class MLP {

    double[][,] w;
    double[][] theta;
    int D; //Depth of MLP
    int[] M; //size of each Layer

    //parameter for calculation
    double[][] a, o;
    double[][] delta;
    double[][,] dw;
    double[][] dth;
    Random rnd;


    double eta =0.001;
    double l1 = 0.000; //Lasso
    double l2 =0.00; //Ridge


    public MLP(int D, int[] M){
      rnd = new Random(1);
      this.D = D;
      this.M = new int[D+1];
      for(int i=0;i<D+1;i++) this.M[i] = M[i];
      this.a = new double[D][];
      for(int i=0;i<D;i++) this.a[i] = new double[this.M[i+1]];
      this.o = new double[D+1][];
      for(int i=0;i<D+1;i++) this.o[i] = new double[this.M[i]];
      this.delta = new double[D][];
      for(int i=0;i<D;i++) this.delta[i] = new double[M[i+1]];
      this.dw = new double[D][,];
      for(int i=0;i<D;i++) this.dw[i] = new double[M[i],M[i+1]];
      this.dth = new double[D][];
      for(int i=0;i<D;i++) this.dth[i] = new double[M[i+1]];

      //Initialization of parameters
      this.w = new double[D][,];
      for(int i=0;i<D;i++){
        this.w[i] = new double[M[i],M[i+1]];
        for(int j=0;j<M[i];j++){
          for(int k=0;k<M[i+1];k++){
            this.w[i][j,k] = rnd.NextDouble()-0.5;
          }
        }
      }
      this.theta = new double[D][];
      for(int i=0;i<D;i++){
        this.theta[i] = new double[M[i+1]];
        for(int j=0;j<M[i+1];j++){
          this.theta[i][j] = rnd.NextDouble()-0.5;
        }
      }
    }

    double sig(double x){ return 1.0/(1.0+Math.Exp(-x)); }
    double sign(double x){ return x>0 ? 1.0: -1.0; }
    double sig_grad(double x){ return sig(x) * (1.0-sig(x)); }
    public void predict(double[] x){
      for(int j=0;j<M[0];j++) o[0][j] = x[j];
      for(int i=0;i<D;i++){
        for(int kn=0;kn<M[i+1];kn++){
          a[i][kn] = theta[i][kn];
          for(int kb=0;kb<M[i];kb++) a[i][kn] += o[i][kb] * w[i][kb,kn];
          o[i+1][kn] = sig(a[i][kn]);
        }
      }
    }
    public double[] Predict(double[] x){
      predict(x);
      double[] ans = new double[M[D]];
      for(int i=0;i<M[D];i++) ans[i] = o[D][i];
      return ans;
    }

    void gradient(double[] x, double[] y){
      predict(x);
      for(int k=0;k<M[D];k++){
        delta[D-1][k] = (o[D][k] - y[k]) * o[D][k] * (1.0-o[D][k]);
      }
      for(int i=D-2;i>-1;i--){
        for(int k=0;k<M[i+1];k++){
          delta[i][k] = 0.0;
          for(int l=0;l<M[i+2];l++) delta[i][k] += w[i+1][k,l] * delta[i+1][l];
          delta[i][k] = delta[i][k] * o[i+1][k] * (1.0-o[i+1][k]);
        }
      }
      for(int i=0;i<D;i++){
        for(int k=0;k<M[i+1];k++){
          dth[i][k] =  delta[i][k];
          for(int j=0;j<M[i];j++){
            dw[i][j,k] =  delta[i][k] * o[i][j];
          }
        }
      }
    }

    public void Check(double[] x, double[] y){
      gradient(x,y);
      double epsilon = 0.00001;
      double[][,] ddw = new double[D][,];
      for(int i=0;i<D;i++) ddw[i] = new double[M[i],M[i+1]];

      double[] bw,ew;
      for(int i=0;i<D;i++){
        for(int j=0;j<M[i];j++){
          for(int k=0;k<M[i+1];k++){
            double defw = w[i][j,k];
            w[i][j,k] = defw + epsilon;
            bw = Predict(x);
            w[i][j,k] = defw - epsilon;
            ew = Predict(x);
            w[i][j,k] = defw;

            double error1 = 0 , error2 = 0;
            for(int s=0;s<M[D];s++){
              error1 += 0.5 * (bw[s]-y[s])*(bw[s]-y[s]);
              error2 += 0.5 * (ew[s]-y[s])*(ew[s]-y[s]);
            }
            ddw[i][j,k] = (error1-error2)/(2*epsilon);
          }
        }
      }

      double sm1 = 0, sm2 = 0, sm3 = 0;
      for(int i=0;i<D;i++){
        for(int j=0;j<M[i];j++){
          for(int k=0;k<M[i+1];k++){
            sm1 += (ddw[i][j,k]-dw[i][j,k])*(ddw[i][j,k]-dw[i][j,k]);
            sm2 += ddw[i][j,k] * ddw[i][j,k];
            sm3 += dw[i][j,k] * dw[i][j,k];
          }
        }
      }
      double an = Math.Sqrt(sm1)/(Math.Sqrt(sm2)+Math.Sqrt(sm3)+epsilon*epsilon);
      Console.WriteLine("Accuracy:"+an.ToString());
    }

    public void train(double[] x, double[] y){
      gradient(x,y);
      for(int i=0;i<D;i++){
        for(int k=0;k<M[i+1];k++){
          theta[i][k] -= eta * (dth[i][k]+l1*sign(theta[i][k])+l2*theta[i][k]);
          for(int j=0;j<M[i];j++){
            w[i][j,k] -= eta * (dw[i][j,k]+l1*sign(w[i][j,k])+l2*w[i][j,k]);
          }
        }
      }
    }
  }
}
