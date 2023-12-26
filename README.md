---
runme:
  id: 01HH2HE8VJT371W3SGQGJPSDPR
  version: v2.0
---

# Kriging Model Repository

Using Kriging Model for basic understanding and application in Python for spacial statistics. Tutorial followed is:

- [Kriging Model Tutorial](https://youtu.be/vhVDcCNNMWE?si=BtIr1Pxi41Z1Yp-l)
- [Basic Idea of Kriging Model](https://youtu.be/J-IB4_QL7Oc?si=XMu8-jrgGClVrupQ)

## Math Understanding

Context Example: Suppose we have an island, and for each of the finite points on the island we have collected elevation data. We then inquire about a point that we did not explicitly collect data at. The best way to find the elevation of that point of interest is to use the Krigin Model. Here's a basic run down of the math.

This point $y_{new}$ of interest has 5 ($k$) nearest points near it. We use k-nearest neighbors approach to estimate the unknown elevation. Using the Kriging Model, our predicted elevation at the point $y_{new}$ is a linear combination of the elevations of the k-nearest neighbors.

Our point of interest has the coordinates $(x_i, x_j, y_j)$ where $x_i, x_j$ corresponds to the spacial placement of this point, and $y_j$ is the elevation at that point.

### Model:

- $y_{new} = w^Ty + ɛ_{new} = w_1y_1 + w_2y_2 + ⋯ + w_5y_5+ ɛ_{new}$

If we know what the weight matrix is ($w$), then we can plug in the known elevations to infer about the unknown elevations. This means we must find the weight matrix. However, we do know the distances between the k-nearest neighbors, let's proceed with the assumption that the closer the distance is to our point, the higher weight we assign. Use variogram to know this.

### Variogram:

Variogram is put simply, the distance between two points ($h$), and the difference in their elevations ($γ(h)$):

$γ(x_i, x_j) = \frac{1}{2}(y_i - y_j)^2$

- Where $y_i,y_j$ are the elevations. $γ$ finding mean the elevation in those two points $\frac{1}{2}$, expecting the closer the points, the smaller the value of $γ$.

![Variogram Image](https://vsp.pnnl.gov/help/image/Variogram.gif)

- The higher the distance $h$ (x-axis), the higher the $γ(h)$ (y-axis).
- Actual variogram better resembles a cloud instead of just a curve.
- The higher the `nugget`, the higher the noise ($\varepsilon$).
- `sill` is the ceiling of the variogram, and when the variogram hits the sill this is valled the `range`.

Solving for the weights ($\hat{w} = \vec{w}$ vectors $i = 1, \cdots, k$), is solving the system of equations $A=\hat{w}b \Rightarrow \hat{w}=A^{-1}b$

- $\hat{w}$ is the predicted weights.
- In this scenario $γ(x_i, x_j) = A$ and $γ(x_{new}, x_i) = \vec{b}$

### When you SHOULD use the Kriging Model

Assumptions:

1. Stationary: (same in time series data) if we look at any small segment of the island, we should get the same attributes such as the mean, standard deviation (volitility), etc
2. Constant variogram: the variogram should look very similar no matter which segment we analyze.

   - Recall: the variogram is the distance between two points ($h$), and the difference in their elevations ($γ(h)$):

Then we are safe to use this model. Transformations of the data are acceptable to do in order to use this model.

### Pros & Cons:

- Pro: The error ($\varepsilon$) in the model gives you the estimated error in your prediction
- Con: This process is computationally intensive. Since every time we run this, we need to solve the system of equations ever time that the elevation changes, or point changes. This is good for a couple points at most, after that this becomes too computationally expensive.

---

### Main Idea of Python Implementation

- Assumptions: spherical model, already obtained the variogram (nugget, range, and sill).
- Know the (x,y) coordinate but want to estimate the best the grade/elevation at this point, using ordinary Kirging.

## Math Background of Python Program

- Calculate the distances between all of the points, as well as the unknown value $P5$, this comprises the matrix $A$. In this matrix, we have the indices of an example element $γ(h_{13})$ is the difference in point 3 and point 1.
- Along the $\operatorname{diag}(A)$ the distance between $γ_1 - γ_1 = 0$.

$$
\begin{pmatrix}
γ(h_{11}) & γ(h_{12}) & γ(h_{13}) & 1\\
γ(h_{21}) & γ(h_{22}) & γ(h_{23}) & 1\\
γ(h_{31}) & γ(h_{32}) & γ(h_{33}) & 1\\
1 & 1 & 1 & 1
\end{pmatrix}
\begin{pmatrix}
e_1\\ e_2\\ e_3\\ λ
\end{pmatrix}
$$

\begin{pmatrix}
\gamma(h_{1P})\\
\gamma(h_{2P})\\
\gamma(h_{3P})\\
1 \\
\end{pmatrix}
$$

Note that from the system of equations above, we denote:

$A = \begin{pmatrix}
γ(h_{11}) & γ(h_{12}) & γ(h_{13}) & 1\\
γ(h_{21}) & γ(h_{22}) & γ(h_{23}) & 1\\
γ(h_{31}) & γ(h_{32}) & γ(h_{33}) & 1\\
1 & 1 & 1 & 0
\end{pmatrix},
\hspace{.3cm}
e = \begin{pmatrix}
e_1\\ e_2\\ e_3\\ λ
\end{pmatrix},
\hspace{.3cm}  
b = \begin{pmatrix}
\gamma(h_{1P})\\
\gamma(h_{2P})\\
\gamma(h_{3P})\\
1 \\
\end{pmatrix}$

In order to solve:  
$$Ae=b ⇒ e = A^{-1}b$$

### Distance Matrix

In 2 dimensions: $D = \sqrt{[(x_2-x_1)^2 + (y_2-y_1)^2]}$

We obtain the distance matrix of :
$$\begin{pmatrix}
X|Y & P1 & P2 & P3 & P4\\
P1 & 0 & D_{12} &  D_{13} & D_{14}\\
P2 & D_{21} & 0 &  D_{23} & D_{24}\\
P3 & D_{31} &  D_{32} & 0 & D_{34}\\
P4 & D_{41} &  D_{42} & D_{34} & 0 \\
\end{pmatrix}
\hspace{.5cm}
\begin{pmatrix}
P5 \\ D_{15} \\ D_{25} \\ D_{35} \\ D_{45}
\end{pmatrix}
$$

Where $NA$ and first column and row of matrix ($P1 - P4$) are used as indexing points.

For each element in the matrix we apply $γ(h) = C0 + C\left( \frac{3}{2} * \frac{h}{a} - \frac{1}{2}\left( \frac{h}{a} \right)^3 \right)$ to this, as such:

- $C0$ = Nugget
- $C$ = Sill
- $a$ = Range
- $h$ = lag distance

The output $γ(h)$ will give us the *semivariance* value for each distance in the matrix below.

$$\begin{pmatrix}
\gamma(0) &  \gamma(D_{12}) &   \gamma(D_{13}) &  \gamma(D_{14})\\
\gamma(D_{21}) &  \gamma(0) &   \gamma(D_{23}) &  \gamma(D_{24})\\
\gamma(D_{31}) &   \gamma(D_{32}) &  \gamma(0) &  \gamma(D_{34})\\
\gamma(D_{41}) &   \gamma(D_{42}) &  \gamma(D_{34}) &  \gamma(0) \\
\end{pmatrix}
\hspace{.5cm}$$

Now to find the solution, we need to solve the following equation to find the weight vector, now denoted as $\hat{e} = \vec{e}$

$$e = \hat{e} = \begin{pmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\ w_5
\end{pmatrix} A^{-1} *
\begin{pmatrix}
\gamma(D_{15}) \\  \gamma(D_{25}) \\  \gamma(D_{35}) \\  \gamma(D_{45})
\end{pmatrix}
$$

$$
\left[
\begin{pmatrix}
\gamma(0) &  \gamma(D_{12}) &   \gamma(D_{13}) &  \gamma(D_{14})\\
\gamma(D_{21}) &  \gamma(0) &   \gamma(D_{23}) &  \gamma(D_{24})\\
\gamma(D_{31}) &   \gamma(D_{32}) &  \gamma(0) &  \gamma(D_{34})\\
\gamma(D_{41}) &   \gamma(D_{42}) &  \gamma(D_{34}) &  \gamma(0) \\
\end{pmatrix}
\right]^{-1}*
\begin{pmatrix}
\gamma(D_{15}) \\  \gamma(D_{25}) \\  \gamma(D_{35}) \\  \gamma(D_{45})
\end{pmatrix}
$$

We obtain the solution by calculating the linear combintation of this model,

$$y_{new} = w^Ty + ɛ_{new} = w_1y_1 + w_2y_2 + ⋯ + w_5y_5+ ɛ_{new}$$

However, for our use this will look something like this:

$$Z(P_5) = W_1Z(P_1) + W_2Z(P_2) + W_3Z(P_3) + W_4Z(P_4)$$

### Image Output

- The images outputted from `main.py` displays of sample distributions with distinct points that represent sample locations from the loaded R data, and in the context of the data these are soil related to their zinc concentrations. The second image is a contoured interpolation over a geographical area, showing variations in zinc concentration across the surface. 
