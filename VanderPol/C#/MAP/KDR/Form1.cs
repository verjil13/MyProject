using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;

namespace MAP_
{
    public partial class Form1 : Form
    {
        double xmin, xmax, ymin, ymax, x0, y0, E, m, w, L, x, y,ac ;
        double h = (2 * Math.PI) / (40);

        int n, n1;
        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void radioButton9_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton9.Checked == true)
            {
                textBox7.Text = "50"; textBox8.Text = "50";
                textBox11.Visible = true; label8.Visible = true; return;
                
            }
            if (radioButton9.Checked == false)
            {
                textBox7.Text = "5000"; textBox8.Text = "70";
                textBox11.Visible = false; label8.Visible = false; return;

            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public Form1()
        {
            InitializeComponent();
        }

        double F(double y)
        {
            return y;
        }
        double G(double x, double y, double E, double m)
        {
            return (E - m * x * x) * y - x;
        }

        public void Runge(ref double x, ref double y, double E, double m, double h)
        {            
            double k1 = F(y);
            double l1 = G(x, y, E, m);
            double k2 = F(y + h * l1 / 2);
            double l2 = G(x + h * k1 / 2, y + h * l1 / 2, E, m);
            double k3 = F(y + h * l2 / 2);
            double l3 = G(x + h * k2 / 2, y + h * l2 / 2, E, m);
            double k4 = F(y + h * l3);
            double l4 = G(x + h * k3, y + h * l3, E, m);
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
            y = y + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6;
        }


        void Imp(ref double x, ref double y, double E, double m, double L, double h)
        {
            for (int j = 1; j <= 10; j++)
            {
                if ((Math.Abs(x) < 100) && (Math.Abs(y) < 100)) Runge(ref x, ref y, E, m, h);
            }
            if (radioButton4.Checked == true) y = y + L * (1 - x * x / 2);
            if (radioButton5.Checked == true) y = y + L * (1 - x * x / 2 + Math.Pow(x, 4) / 24);
            if (radioButton3.Checked == true) y = y + L * (1 - x * x / 2 + Math.Pow(x, 4) / 24 - Math.Pow(x, 6) / 720);
            if (radioButton7.Checked == true) y = y + L * (1 - x * x / 2 + Math.Pow(x, 4) / 24 - Math.Pow(x, 6) / 720+ Math.Pow(x, 8) / 40320);
            if (radioButton10.Checked == true) y = y + L * (1 - x * x / 2 + Math.Pow(x, 4) / 24 - Math.Pow(x, 6) / 720 + Math.Pow(x, 8) / 40320- Math.Pow(x, 10) / 3628800);
            if (radioButton6.Checked == true) y = y + L * Math.Cos(x);
        }


        public void Point(double x, double y, int p)
        {
            int fx, fy; Color c = Color.Navy;
            fx = Convert.ToInt32(Math.Round((x - xmin) * pictureBox1.Width / Math.Abs(xmax - xmin)));
            fy = Convert.ToInt32(Math.Round(pictureBox1.Height - (y - ymin) * pictureBox1.Height / Math.Abs(ymax - ymin)));

            if (p == 1) c = Color.Navy; if (p == 2) c = Color.Green; if (p == 3) c = Color.Blue;
            if (p == 4) c = Color.Maroon; if (p == 5) c = Color.Olive; if (p == 6) c = Color.Lime;
            if (p == 7) c = Color.Teal; if (p == 8) c = Color.Purple; if (p == 9) c = Color.Yellow;
            if (p == 10) c = Color.Aqua; if (p == 11) c = Color.Silver; if (p == 12) c = Color.Gray;
            if (p == 13) c = Color.Fuchsia; if (p == 14) c = Color.Red; if (p == 15) c = Color.SkyBlue;
            Graphics g = pictureBox1.CreateGraphics();
            if (p < 16) { g.FillRectangle(new SolidBrush(c), new Rectangle(fx, fy, 1, 1)); }
            if (p >= 16 && p < 25) { g.FillRectangle(new SolidBrush(Color.Gray), new Rectangle(fx, fy, 1, 1)); }
            if (p >= 25) { g.FillRectangle(new SolidBrush(Color.Black), new Rectangle(fx, fy, 1, 1)); }
        }
        public void Point1(double x, double y, double l1)
        {
            int fx, fy; Color c = Color.Navy; int t = 0;
            fx = Convert.ToInt32(Math.Round((x - xmin) * pictureBox1.Width / Math.Abs(xmax - xmin)));
            fy = Convert.ToInt32(Math.Round(pictureBox1.Height - (y - ymin) * pictureBox1.Height / Math.Abs(ymax - ymin)));
            if (l1 > ac) c = Color.Black;
            else
            if (l1 < -ac)
            {
                if (255+255 * l1 / 0.5 > 240) t = 240; else
                if (255+255 * l1 / 0.5 < 80) t = 80; else
                t= Convert.ToInt32(255+255 * l1 / 0.5);
                c = Color.FromArgb(t, t, t);

            }

            else
                c = Color.Red;
            Graphics g = pictureBox1.CreateGraphics();
            g.FillRectangle(new SolidBrush(c), new Rectangle(fx, fy, 1, 1));
        }

        void SS(ref double x, ref double y, double E, double m, double L)
        {
            int p;
            double atx, aty;
           // if ((Math.Abs(x) > 100) || (Math.Abs(y) > 100)) { x = x0; y = y0; }
            x = x0; y = y0;
            for (int i = 1; i < n; i++)
            {
                if ((Math.Abs(x) > 100) && (Math.Abs(y) > 100)) 
                    break;
                Imp(ref x, ref y, E, m, L, h);
            }
            atx = x; aty = y;
            p = 0;
            for (int i = 1; i < n1; i++)
            {
                if ((Math.Abs(x) > 100) || (Math.Abs(y) > 100)) 
                    break;
                
                Imp(ref x, ref y, E, m, L, h);
                p = p + 1;
                if (Math.Sqrt((atx - x) * (atx - x) + (aty - y) * (aty - y)) < 0.000000001) 
                    break;                
               
            }
            if ((Math.Abs(x) < 100) && (Math.Abs(y) < 100)) Point(L, E, p);
        }

        void LL(ref double x, ref double y, double E, double m, double L)
        {
              double l1 = 0;
              if ((Math.Abs(x) > 100) || (Math.Abs(y) > 100))
              { x = x0; y = x0; }
              for (int i = 1; i < 5000; i++) if ((Math.Abs(x) < 100) && (Math.Abs(y) < 100)) { Imp(ref x, ref y, E, m, L, h); }
              double s1 = 0; 
              for (int i = 1; i < n; i++)
              {
                  double dx = x + 0.00001, dy = y + 0.00001;
                  for (int j = 1; j < n1; j++)  if ((Math.Abs(x) < 100) && (Math.Abs(y) < 100))
                  {
                      Imp(ref x, ref y, E, m, L, h);
                      Imp(ref dx, ref dy, E, m, L, h);
                  }
                  s1 = s1 + Math.Log(Math.Sqrt((x - dx) * (x - dx) + (y - dy) * (y - dy)) / (0.00001* Math.Sqrt(2)));
              }
              l1 = s1 / (n * n1);
              if ((Math.Abs(x) < 100) && (Math.Abs(y) < 100)) Point1(L, E, l1);
           
        }

        public void button1_Click(object sender, EventArgs e)
        {
            
            Graphics g1 = pictureBox1.CreateGraphics();
            DateTime d1 = DateTime.Now;
            g1.Clear(Color.White);
            if (radioButton8.Checked == true) g1.FillRectangle(new SolidBrush(Color.White), new Rectangle(0, 0, pictureBox1.Width, pictureBox1.Height));
            if (radioButton9.Checked == true) g1.FillRectangle(new SolidBrush(Color.Green), new Rectangle(0, 0, pictureBox1.Width, pictureBox1.Height));
            xmin = Convert.ToDouble(textBox1.Text);  xmax = Convert.ToDouble(textBox2.Text);
            ymin = Convert.ToDouble(textBox3.Text);  ymax = Convert.ToDouble(textBox4.Text);
            n = Convert.ToInt32(textBox7.Text);      n1 = Convert.ToInt32(textBox8.Text);
            x0 = Convert.ToDouble(textBox5.Text);    y0 = Convert.ToDouble(textBox6.Text);      
            x = x0; y = y0;
            double dL = Math.Abs(xmax - xmin) / pictureBox1.Width, dE = Math.Abs(ymax - ymin) / pictureBox1.Height;
            m = Convert.ToDouble(textBox9.Text); w = Convert.ToDouble(textBox10.Text);
            ac = Convert.ToDouble(textBox11.Text);
            double[] EE = new double[pictureBox1.Height], xx = new double[pictureBox1.Height], yy = new double[pictureBox1.Height]; int rr;
            rr = 0;
            if (radioButton2.Checked==true)
            for (int i = 0; i < pictureBox1.Height; i++)
            {
                EE[i] = ymin + i * dE;
                xx[i] = 0.1; yy[i] = 0.1;
            }
            L = xmin;
            while (L <= xmax)
            {
                if (radioButton1.Checked == true)
                {
                    E = ymin;
                    while (E <= ymax)
                    {
                        if (radioButton8.Checked == true) SS(ref x, ref y, E, m, L);
                        if (radioButton9.Checked == true) LL(ref x, ref y, E, m, L);
                        E = E + dE;
                    }
                }
                if (radioButton2.Checked == true)
                    Parallel.For(0, pictureBox1.Height, i =>
                    {
                        
                        if (radioButton8.Checked == true) SS(ref xx[i], ref yy[i], EE[i], m, L);
                         if (radioButton9.Checked == true) LL(ref xx[i], ref yy[i], EE[i], m, L);
                    });                
                L = L + dL;
                if ((Math.Abs(L) < dL) && (rr == 1)) { L = 0; rr = 1; }
            }
            
           
            DateTime d2 = DateTime.Now;
            label7.Text = "Ready "+ Convert.ToString((d2 - d1).TotalSeconds) + " sec";
        }
    }
}
