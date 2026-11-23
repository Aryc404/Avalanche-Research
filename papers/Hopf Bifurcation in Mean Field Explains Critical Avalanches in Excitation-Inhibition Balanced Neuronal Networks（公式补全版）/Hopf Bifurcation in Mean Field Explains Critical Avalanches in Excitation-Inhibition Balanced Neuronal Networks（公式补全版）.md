# 1.文中使用的网络模型

**Current-based LIF 神经元网络，也就是电流型漏积分发放网络。**

这篇文章主要研究：单个神经元一直不规则地放电时，整个网络为什么仍然能出现振荡和临界雪崩？

网络中一共包含10,000个神经元，其中 **E神经元8,000个，I神经元2,000个**，比例为4:1。

神经元之间随机连接，连接概率为 **20%**。包含E→E、E→I、I→E、I→I四类递归突触，平均每个神经元接收1,600个E来源和400个I来源的递归输入。

每个神经元还接收1,600路独立的兴奋性Poisson输入，单路频率记为 $Q_o$。它模拟来自其他回路的背景驱动，主图常用5 Hz；部分实验会改变这一频率。

**网络连接和突触权重在模拟中固定，没有STDP、STP或自适应阈值。** 作者主要通过改变抑制性突触的衰减时间 $\tau_d^I$，观察网络如何从异步活动转变为振荡，并寻找临界雪崩所在的参数区域。[^model]

## 1.1 论文中使用的核心公式：单个神经元与突触

* **膜电位变化方程（原文式1）**

$$
\begin{aligned}
\frac{dV_i}{dt}={}&f_\alpha(V_i)
+J_{\alpha o}\sum_{j\in\partial_i^o}(F^E*s_j)(t-\tau_l^E)\\
&+J_{\alpha E}\sum_{j\in\partial_i^E}(F^E*s_j)(t-\tau_l^E)
+J_{\alpha I}\sum_{j\in\partial_i^I}(F^I*s_j)(t-\tau_l^I),\\
f_\alpha(V_i)={}&\frac{V_{\rm rest}^\alpha-V_i}{\tau_\alpha},
\qquad s_j(t)=\sum_n\delta(t-t_j^n).
\end{aligned}
$$

其中：

- $V_i$：第 $i$ 个神经元的膜电位；$\alpha=E,I$ 表示它属于哪一类。
- $f_\alpha(V_i)$：漏电项，使膜电位回到静息电位；$V_{\rm rest}^E=V_{\rm rest}^I=-70\,\mathrm{mV}$。
- $\tau_E=20\,\mathrm{ms}$、$\tau_I=10\,\mathrm{ms}$：E、I神经元的膜时间常数。
- $\partial_i^o,\partial_i^E,\partial_i^I$：连接到神经元 $i$ 的外部、网络内E、网络内I来源集合。
- $s_j(t)$：神经元 $j$ 的脉冲序列；$t_j^n$ 是它第 $n$ 次发放的时间，$\delta$ 为Dirac脉冲。
- $F^E,F^I$：兴奋性、抑制性突触滤波核；$*$ 表示卷积，即把一个个瞬时spike变成具有时间过程的输入。
- $\tau_l^E,\tau_l^I$：突触传导延迟。本文主结果均取0。
- $J_{\alpha\beta}$：$\beta$ 类来源对 $\alpha$ 类目标的固定权重。**第一个下标是目标，第二个下标是来源。**

按这个下标约定，权重为：

$$
\begin{aligned}
J_{Eo}&=0.45\,\mathrm{mV},&J_{Io}&=0.72\,\mathrm{mV},\\
J_{EE}&=0.36\,\mathrm{mV},&J_{IE}&=0.72\,\mathrm{mV},\\
J_{EI}&=-0.81\,\mathrm{mV},&J_{II}&=-1.44\,\mathrm{mV}.
\end{aligned}
$$

对于公式的理解：第一项是膜电位自行回落；第二项是外部背景驱动；第三项是网络内部兴奋性输入；第四项是网络内部抑制性输入。输入的正负由 $J$ 决定。这里的突触输入没有乘上 $(E_{\rm reversal}-V_i)$，因此主模型属于电流型LIF。

当膜电位达到阈值 $V_{\rm th}=-50\,\mathrm{mV}$ 时，神经元发放一个spike，随后重置到 $V_{\rm reset}=-60\,\mathrm{mV}$。E、I神经元的不应期分别为2 ms和1 ms，不应期内暂停突触积分。[^model]

* **双指数突触核（原文式2）**

$$
F^\alpha(t)=\frac{1}{\tau_d^\alpha-\tau_r}
\left[\exp\left(-\frac{t}{\tau_d^\alpha}\right)
-\exp\left(-\frac{t}{\tau_r}\right)\right],\qquad t\geq0.
$$

其中：

- $\tau_r=0.5\,\mathrm{ms}$：突触输入的上升时间常数，E和I均相同。
- $\tau_d^E=2\,\mathrm{ms}$：兴奋性突触衰减时间常数。
- $\tau_d^I=1\text{–}4.5\,\mathrm{ms}$：抑制性突触衰减时间常数，也是本文主要扫描的控制参数。
- $\alpha$ 表示突触前神经元的类型，因此外部兴奋性输入也使用 $F^E$。

一个spike到达后，突触作用先上升，再逐渐下降。**增大 $\tau_d^I$，改变的是抑制作用消退的速度；它不等于改变I神经元的膜时间常数 $\tau_I$。**

双指数核经过归一化，$\int_0^\infty F^\alpha(t)dt=1$。因此改变其时间尺度会改变输入在时间上的分布，并不改变单次脉冲核的总面积。[^model]

* **双指数滤波的等价微分形式（原文式3）**

$$
\left(\tau_d^\alpha\frac{d}{dt}+1\right)
\left(\tau_r\frac{d}{dt}+1\right)
\left[(F^\alpha*s_j)(t-\tau_l^\alpha)\right]
=\sum_n\delta(t-t_j^n-\tau_l^\alpha).
$$

其中两项微分算子分别对应衰减和上升过程，右侧是延迟后到达的突触前spike。它与前面的卷积表达描述同一件事：**输入脉冲经过两个时间过程，形成双指数响应。** 作者使用这个形式推导平均场方程。[^meanfield]

模拟步长为0.05 ms，采用修正的二阶Runge-Kutta方法。每个参数条件运行16 s，丢弃前1 s，再对15次随机初始膜电位的结果平均。[^model]

## 1.2 由单个神经元推到整个网络：平均场公式

这里的“平均场”可以理解为：不再逐个追踪10,000个神经元，而是追踪E、I两个群体的平均膜电位和平均突触输入。

先定义：

$$
V_\alpha(t)=\langle V_i(t)\rangle_{i\in\alpha},\qquad
n_\alpha=pN_\alpha,
$$

$$
\Phi_\alpha(t)=
\left\langle\sum_{j\in\partial_i^\alpha}
(F^\alpha*s_j)(t-\tau_l^\alpha)\right\rangle_{i\in E,I}.
$$

其中：$N_\alpha$ 是该类神经元总数；$n_\alpha$ 是平均每个神经元接收的该类来源数量；$\Phi_\alpha$ 是平均每个目标神经元收到的该类滤波后输入。尖括号表示对相应神经元群体取平均。

* **对突触输入取群体平均（原文式4、5）**

$$
\begin{aligned}
&\left(\tau_d^\alpha\frac{d}{dt}+1\right)
\left(\tau_r\frac{d}{dt}+1\right)\Phi_\alpha(t)\\
&\quad=
\left\langle\sum_{j\in\partial_i^\alpha}\sum_n
\delta(t-t_j^n-\tau_l^\alpha)\right\rangle_{i\in E,I}
\approx n_\alpha Q_\alpha(t-\tau_l^\alpha).
\end{aligned}
$$

其中第一处等号对应式4，最后的近似对应式5；$Q_\alpha$ 为该类神经元的群体平均放电率。

理解：平均每个神经元有 $n_\alpha$ 个该类输入来源，每个来源的平均放电率是 $Q_\alpha$，因此总的平均脉冲输入近似为 $n_\alpha Q_\alpha$。这个近似依赖大规模、均匀随机网络的条件。[^meanfield]

* **群体放电率的定义（原文式6）**

$$
Q_\alpha(t)=\lim_{\Delta t\to0}\frac{1}{\Delta t}
\int_t^{t+\Delta t}
\left\langle\sum_n\delta(s-t_j^n)\right\rangle_{j\in\alpha}ds.
$$

其中：$\Delta t$ 是计数时间窗；积分统计窗内spike数；先对神经元取平均，再除以时间窗，就得到单位时间、每个神经元的平均发放次数。

实际统计采用 $\Delta t=1\,\mathrm{ms}$。在这个约定下，一个神经元在窗内至多发放一次，群体内发放神经元的比例，其数值可以对应每毫秒的平均放电率。换成Hz时需要乘1,000，不能把后面sigmoid的上限1直接理解为1 Hz。[^meanfield]

* **从膜电位超过阈值的比例估计放电率（原文式7、8）**

$$
Q_\alpha(t)=\left\langle H(V_j-V_{\rm th})\right\rangle_{j\in\alpha},
\qquad
H(x)=\begin{cases}1,&x\geq0,\\0,&x<0.\end{cases}
$$

这里的 $V_j$ 指数值更新时、执行重置前的膜电位。若把同一时刻群体膜电位近似为均值 $V_\alpha(t)$、方差 $\sigma_\alpha^2$ 的高斯分布 $P_\alpha(V)$，就得到：

$$
\begin{aligned}
Q_\alpha(t)
&=\int_{V_{\rm th}}^\infty P_\alpha(V)dV\\
&=\frac12-\frac12\operatorname{erf}
\left(\frac{V_{\rm th}-V_\alpha(t)}{\sqrt2\,\sigma_\alpha}\right),
\qquad
\operatorname{erf}(x)=\frac{2}{\sqrt\pi}\int_0^x e^{-u^2}du.
\end{aligned}
$$

其中：$H$ 是阶跃函数；$\operatorname{erf}$ 是误差函数；$\sigma_\alpha$ 控制膜电位分布的宽度。

直观上，平均膜电位越接近阈值，或者膜电位分布越宽，超过阈值的神经元比例就可能越高。这是把平均膜电位与群体发放联系起来的步骤，使用的是本文1 ms计数约定。[^meanfield]

* **将误差函数近似为sigmoid（原文式9）**

利用 $\operatorname{erf}(x)\approx\tanh(\pi x/\sqrt6)$，作者得到：

$$
\begin{aligned}
Q_\alpha(t)
&\approx\frac12-\frac12\tanh
\left(\frac{\pi}{\sqrt6}
\frac{V_{\rm th}-V_\alpha(t)}{\sqrt2\,\sigma_\alpha}\right)\\
&=\frac{1}{1+\exp\left[
\frac{\pi(V_{\rm th}-V_\alpha(t))}{\sqrt3\,\sigma_\alpha}\right]}.
\end{aligned}
$$

符号含义与式8相同。这条S形曲线是平均场的非线性“输入—输出关系”：用平均膜电位估计发放水平。**它不把微观LIF模型的硬阈值换成随机发放规则，只是宏观统计近似。**[^meanfield]

* **有效参数 $\sigma_\alpha$ 的估计（原文式10）**

$$
\sigma_\alpha=
\frac{V_{\rm th}-V_\alpha^{\rm ss}}
{\ln\left[(Q_\alpha^{\rm ss})^{-1}-1\right]}
\frac{\pi}{\sqrt3}.
$$

其中：$V_\alpha^{\rm ss}$ 和 $Q_\alpha^{\rm ss}$ 是微观网络异步稳态下测得的平均膜电位和放电率，$\mathrm{ss}$ 表示steady state。$Q_\alpha^{\rm ss}$ 必须使用前述每毫秒的数值约定。

作者先从微观模拟得到这两个量，再反推出 $\sigma_\alpha$，因此称为**半解析平均场**。这个有效参数也吸收了不应期、相关性及其他近似误差的影响，不能简单当作直接测量的膜电位标准差。[^meanfield]

* **平均突触输入的动力学（原文式11）**

$$
\left(\tau_d^\alpha\frac{d}{dt}+1\right)
\left(\tau_r\frac{d}{dt}+1\right)\Phi_\alpha(t)
=\frac{n_\alpha}
{1+\exp\left[
\frac{\pi\{V_{\rm th}-V_\alpha(t-\tau_l^\alpha)\}}
{\sqrt3\,\sigma_\alpha}\right]},
\qquad\alpha=E,I.
$$

右侧就是 $n_\alpha Q_\alpha(t-\tau_l^\alpha)$。因此，这条公式表示“群体平均膜电位→群体放电→经过突触时间过程的平均反馈输入”。主结果中传导延迟取0，但仍保留突触上升和衰减时间。[^meanfield]

* **群体平均膜电位方程（原文式12）**

$$
\begin{aligned}
\frac{dV_\alpha}{dt}
={}&\frac{V_{\rm rest}^\alpha-V_\alpha}{\tau_\alpha}
+J_{\alpha o}\left(n_oQ_o+
\sqrt{\frac{n_oQ_o}{N_\alpha}}\,\xi_\alpha(t)\right)\\
&+J_{\alpha E}\Phi_E+J_{\alpha I}\Phi_I,
\qquad\alpha=E,I.
\end{aligned}
$$

其中：

- $n_oQ_o$：每个细胞外部Poisson输入的平均强度，$n_o=1,600$。
- $\xi_E(t),\xi_I(t)$：两路相互独立的标准高斯白噪声。
- $\sqrt{n_oQ_o/N_\alpha}$：群体平均后噪声的幅度，包含有限网络规模效应。
- $\Phi_E,\Phi_I$：由式11决定的递归兴奋、抑制输入。

式11与式12共同构成网络的宏观方程。数值模拟宏观波动时保留噪声；分析固定点稳定性时令噪声为0。**微观模型的外部输入是经过突触滤波的Poisson脉冲，宏观式12中的白噪声属于扩散近似。**[^meanfield]

## 1.3 用平均场判断固定点与Hopf分岔

* **稳态方程（原文式13）**

$$
\frac{V_{\rm rest}^\alpha-V_\alpha}{\tau_\alpha}
+J_{\alpha o}n_oQ_o
+J_{\alpha E}n_EQ_E
+J_{\alpha I}n_IQ_I=0,
\qquad\alpha=E,I.
$$

它来自式11、12在无噪声条件下令全部时间导数为0，此时 $\Phi_\alpha=n_\alpha Q_\alpha$。

理解：稳态时，漏电、背景驱动、递归兴奋与递归抑制相互抵消；$Q_E,Q_I$ 则通过式9与 $V_E,V_I$ 相联系。由于突触核面积归一化，$\tau_d^\alpha,\tau_r$ 不直接出现在这条稳态方程中；它们仍会改变该稳态是否稳定。[^stability]

* **大规模极限下的E-I平衡条件（原文式14）**

$$
J_{\alpha o}n_oQ_o
+J_{\alpha E}n_EQ_E
+J_{\alpha I}n_IQ_I=0,
\qquad\alpha=E,I.
$$

在 $n_o,n_E,n_I$ 随 $N$ 增长、权重按 $J\sim N^{-1/2}$ 缩放的强耦合大网络中，式13的漏电项相对较小，可以忽略，得到上述传统平衡网络关系。它主要表达大兴奋输入与大抑制输入的抵消。

原文还给出正稳态解的条件：$\{J_{Eo}/J_{Io},\ J_{EI}/J_{II},\ J_{EE}/J_{IE}\}$ 按顺序递增或递减。本文默认参数满足这一平衡条件。[^stability]

* **固定点的雅可比矩阵（原文式15，以下用系数缩写作等价重写）**

无传导延迟时，取状态向量：

$$
\mathbf X=(V_E,V_I,\Phi_E,\dot\Phi_E,\Phi_I,\dot\Phi_I)^T.
$$

为了清楚呈现6行6列矩阵，记：

$$
A_\alpha=\frac{1}{\tau_d^\alpha\tau_r},\qquad
B_\alpha=\frac{1}{\tau_d^\alpha}+\frac{1}{\tau_r},\qquad
C_\alpha=\frac{n_\alpha Q'_\alpha(V_\alpha)}{\tau_d^\alpha\tau_r}.
$$

则原文矩阵等价写为：

$$
\mathcal J=
\begin{pmatrix}
-1/\tau_E&0&J_{EE}&0&J_{EI}&0\\
0&-1/\tau_I&J_{IE}&0&J_{II}&0\\
0&0&0&1&0&0\\
C_E&0&-A_E&-B_E&0&0\\
0&0&0&0&0&1\\
0&C_I&0&0&-A_I&-B_I
\end{pmatrix},
$$

$$
Q'_\alpha(V_\alpha)=
\frac{\pi\exp\left[\frac{\pi(V_{\rm th}-V_\alpha)}{\sqrt3\sigma_\alpha}\right]}
{\sqrt3\sigma_\alpha\left\{1+
\exp\left[\frac{\pi(V_{\rm th}-V_\alpha)}{\sqrt3\sigma_\alpha}\right]\right\}^2}.
$$

其中：$Q'_\alpha$ 是式9对平均膜电位的导数，在式13求出的固定点处计算；$A_\alpha,B_\alpha,C_\alpha$ 是笔记中为缩短矩阵而定义的系数，均由原文参数确定；$\mathcal J$ 为雅可比矩阵，与突触权重 $J_{\alpha\beta}$ 区分。

对矩阵求特征值：

$$
\lambda_\pm=\rho\pm i\omega,\qquad f_{\rm osc}\approx\frac{\omega}{2\pi}.
$$

这里用 $\rho$ 表示实部，以免与雪崩持续时间指数 $\alpha$ 混淆；这是原文第10页文字所述关系的记号重写。若方程以ms为时间单位，$\omega/(2\pi)$ 的单位是每毫秒，换成Hz需再乘1,000。

当全部特征值实部为负，固定点稳定；当一对主导共轭特征值的实部越过0，就可能发生Hopf分岔。本文随 $\tau_d^I$ 增加，在约3 ms处出现超临界Hopf分岔，稳定固定点转为稳定周期运动。$\omega/(2\pi)$ 给出转变附近振荡频率的线性近似。[^stability]

# 2.测量网络状态与雪崩所用指标

## 2.1 Spike count与群体放电率

把时间分成宽度为 $\Delta t$ 的bin，统计神经元 $i$ 在第 $k$ 个bin内的spike数，记为 $N_i(k)$；再把某一群体的计数相加。

为避免把“群体细胞数”和“某个bin的脉冲数”都写成 $N_\alpha$，以下把群体总计数改记为 $C_\alpha(k)$：

$$
C_\alpha(k)=\sum_{i\in\alpha}N_i(k),\qquad
q_\alpha(k)=\frac{C_\alpha(k)}{N_\alpha\Delta t}.
$$

其中 $N_\alpha$ 是群体神经元数，$q_\alpha$ 为每个细胞的群体平均放电率。原文该处的分母写作 $n_\alpha$；这里按“每个神经元的群体平均率”改写为群体规模 $N_\alpha$，以区别前文表示平均入度的 $n_\alpha$。$C_\alpha$ 也是笔记改记号。[^metrics]

## 2.2 单个神经元的放电是否不规则：ISI的CV

$$
\operatorname{CV}_i=
\frac{\operatorname{std}(\mathrm{ISI}_i)}
{\operatorname{mean}(\mathrm{ISI}_i)}.
$$

其中 $\mathrm{ISI}_i$ 是第 $i$ 个神经元相邻spike之间的间隔；$\operatorname{std}$ 为标准差。

完全规则的发放具有CV=0；Poisson过程CV=1。本文比较不同网络状态下CV的分布，检查群体振荡时单细胞是否仍不规则。CV接近1不能单独证明整个spike序列是Poisson过程。[^metrics]

## 2.3 单细胞与群体放电计数的波动：Fano Factor

$$
\operatorname{FF}_i=
\frac{\operatorname{var}_k[N_i(k)]}{\langle N_i(k)\rangle_k},
\qquad
\operatorname{FF}_\alpha=
\frac{\operatorname{var}_k[C_\alpha(k)]}{\langle C_\alpha(k)\rangle_k}.
$$

其中 $\operatorname{var}_k$ 和 $\langle\cdot\rangle_k$ 表示跨时间bin求方差和均值。本文用 **50 ms计数窗**。

前者量化单细胞随时间的计数波动，后者量化整个群体的计数波动。把两者放在一起，才能区分群体是否形成了比单细胞更显著的集体起伏。[^metrics]

## 2.4 神经元之间是否同步：PCC与膜电位coherence

* **Pearson correlation coefficient，PCC**

先以1 ms分箱，再用50 ms方窗平滑计数序列：

$$
\widetilde N_i(k)=\frac{1}{L}\sum_{s=0}^{L-1}N_i(k-s),\qquad L=50.
$$

再计算：

$$
c_{ij}=
\frac{\operatorname{cov}(\widetilde N_i,\widetilde N_j)}
{\sqrt{\operatorname{var}(\widetilde N_i)\operatorname{var}(\widetilde N_j)}},
\qquad \operatorname{PCC}=\langle c_{ij}\rangle_{i,j\in E}.
$$

其中 $\operatorname{cov}$ 是协方差。作者对E细胞对取平均，以衡量发放事件的同步程度。平滑窗会影响相关系数的数值，因此比较时要保持相同处理流程。[^metrics]

* **膜电位coherence**

每1 ms记录一次膜电位，定义：

$$
\chi_\alpha=
\sqrt{\frac{\operatorname{var}_t[V_\alpha(t)]}
{\left\langle\operatorname{var}_t[V_i(t)]\right\rangle_{i\in\alpha}}},
\qquad V_\alpha(t)=\langle V_i(t)\rangle_{i\in\alpha}.
$$

其中 $\chi_\alpha$ 是笔记给coherence使用的记号，避免与平均场有效参数 $\sigma_\alpha$ 混淆。分子是群体平均膜电位随时间的方差，分母是单细胞膜电位方差的群体平均。

细胞膜电位一起波动时，平均信号仍有明显波动，$\chi_\alpha$ 较大；各自波动相互抵消时，$\chi_\alpha$ 较小。本文主要考察E群体。[^metrics]

## 2.5 群体活动波动与振荡：放电率CV和PSD

$$
\operatorname{CV}_{q_E}=
\frac{\operatorname{std}_t[q_E(t)]}{\langle q_E(t)\rangle_t}.
$$

这里的CV针对群体放电率时间序列，使用1 ms计数窗，和前面单细胞ISI的CV是两个指标。

PSD，即功率谱密度，用于观察群体放电率是否存在明显的振荡频率。原文第11页还用：

$$
P(f)\sim f^{-\beta_{\rm PSD}}
$$

描述谱背景的幂律衰减；$f$ 为频率，$\beta_{\rm PSD}$ 是笔记改记号，避免与雪崩平均大小—持续时间指数混用。Fig. 3D的临界例子在约100 Hz附近有谱峰，并叠加在近似幂律衰减的背景上。**PSD中的幂律与雪崩分布中的幂律分别描述不同统计对象。**[^metrics][^exp5]

## 2.6 文中是怎么定义avalanche的？

**本文实际统计spike雪崩时，使用连续非空bin定义。**

Step 1：合并全部E神经元的spike train，得到按时间排序的spike时刻 $t_1,t_2,\ldots,t_M$，计算：

$$
T_m=\frac{1}{M-1}\sum_{r=1}^{M-1}(t_{r+1}-t_r),\qquad
\Delta t=T_m.
$$

其中 $T_m$ 是合并序列的平均相邻脉冲间隔；它不是“先求每个神经元自己的平均ISI，再对细胞取平均”。以上是原文文字定义的展开写法。

Step 2：用 $\Delta t=T_m$ 对E群体spike分箱。**连续非空bin构成一次雪崩，前后由空bin分隔**；不检查这些spike之间是否存在实际突触连接。

Step 3：如果第 $a$ 次雪崩从第 $k_a$ 个bin持续到第 $\ell_a$ 个bin，按原文定义写成：

$$
S_a=\sum_{k=k_a}^{\ell_a}C_E(k),\qquad
T_a=\ell_a-k_a+1,\qquad
T_a^{\rm time}=T_a\Delta t.
$$

其中：$S_a$ 是spike总数；$T_a$ 是连续非空bin的数量；$T_a^{\rm time}$ 是把持续时间换算为实际时间的笔记记号。原文持续时间主要记作 $T$，以bin数表示。

同一个神经元在雪崩中多次发放，每次都会计入 $S$。一个较长的雪崩也不要求始终由同一批神经元参与。本文称这种方法为 **time-binning、non-causal avalanche**，与基于已知连接追踪前驱spike的因果雪崩不同。[^avalanche]

## 2.7 怎样判断雪崩是否接近临界？

* **第一步：用分布距离 $D$ 粗略定位候选区域**

将雪崩大小频率分布在双对数坐标上拟合直线：

$$
\min_{b_0,b_1}\sum_S
\left[\log_{10}P(S)-(b_0+b_1\log_{10}S)\right]^2,
$$

$$
\log_{10}P(S)\approx b_0+b_1\log_{10}S,
\qquad
P_{\rm fit}(S)=10^{b_0+b_1\log_{10}S}.
$$

$$
D=\frac{\sum_S S\,|P(S)-P_{\rm fit}(S)|}
{\sum_S S\,P(S)}.
$$

其中：$P(S)$ 是实测雪崩大小频率；$b_0,b_1$ 是最小二乘拟合的截距和斜率；$P_{\rm fit}$ 是拟合频率；$D$ 为按雪崩大小加权、再用平均大小归一化的差异。

原文用80个绘图直方图箱估计大小分布，然后计算这个指标。$D$ 越小，表示在该粗略标准下越接近拟合幂律；**$D$ 小本身不构成充分的临界性证据。**[^statistics]

* **第二步：检验大小、持续时间分布及指数之间的关系**

$$
P(S)\sim S^{-\tau},\qquad
P(T)\sim T^{-\alpha},\qquad
\langle S\rangle(T)\sim T^{1/(\sigma\upsilon z)}.
$$

其中：$\tau$ 为大小分布指数；$\alpha$ 为持续时间分布指数；$\langle S\rangle(T)$ 为持续时间相同的雪崩的平均大小。原文把第三个指数记作 $1/(\sigma\upsilon z)$，可整体记为 $\beta$；这里的 $\sigma$ 不等于平均场中的有效膜电位参数 $\sigma_\alpha$。

临界标度关系为（原文式16）：

$$
\boxed{\frac{\alpha-1}{\tau-1}=\frac{1}{\sigma\upsilon z}
\equiv\beta.}
$$

实际做法是分别用分布拟合得到 $\tau,\alpha$，再直接拟合 $\langle S\rangle(T)$ 得到 $\beta_{\rm fit}$，检查其是否接近分布指数预测的 $(\alpha-1)/(\tau-1)$。本文使用NCC toolbox的最大似然估计与双端截断KS检验：截断区间需超过全范围对数跨度的1/3，而且KS检验 $p>0.1$。第三个指数用加权最小二乘估计，并限制在拟合持续时间指数所用的截断范围内。[^statistics]

**本文不把 $\tau=1.5$、$\alpha=2$ 作为必须满足的固定标准。** 外部驱动变化时，指数可以改变，仍需联合检查幂律拟合及指数标度关系。

## 2.8 文中另一个用于机制解释的“宏观雪崩”及其公式

下面对应原文第12–13页的理论解释，与前面实际提取spike雪崩的算法要分开。

* **宏观信号的首次返回定义**

对于群体信号 $x(t)$，如平均放电率，当信号从阈值上方很小的位置开始：

$$
x(0)=x_{\rm th}+\varepsilon,\qquad\varepsilon\to0^+,
$$

到它第一次返回阈值 $x(T)=x_{\rm th}$，定义一次宏观雪崩。$T$ 是首次返回时间，大小为：

$$
S=\int_0^T x(t)dt.
$$

其中 $x_{\rm th}$ 是宏观阈值。**原文这里积分的是 $x(t)$，没有写成减去阈值后的面积 $\int[x(t)-x_{\rm th}]dt$。**[^limits]

* **噪声驱动的Hopf振幅方程（原文式19）**

令 $X(t)=x(t)-x_{\rm th}$，作者启发式地考虑：

$$
\frac{dX}{dt}=aX-X^3+\eta(X,t),
\qquad \eta(X,t)=h+\sqrt X\,\xi(t).
$$

其中：

- $X$：超过阈值的活动量；此处是标量，与前面的六维状态向量 $\mathbf X$ 不同。
- $a$：控制参数；$a=0$ 对应Hopf振幅标准型的分岔点。
- $aX-X^3$：振幅的确定性增长与饱和项。
- $\xi(t)$：标准高斯白噪声。
- $h$：平均偏置，包含递归兴奋、递归抑制与外部输入的影响。
- $\sqrt X$：活动相关噪声的幅度；随着活动量增加，涨落幅度按平方根增长。

理解：在分岔点附近，原本决定活动回落或增长的确定性趋势变弱，噪声会显著影响活动何时回到阈值。这给出了首次返回过程出现无标度统计的一种解释。它是论文引用随机游走理论作出的机制说明，没有逐一还原微观spike雪崩。[^limits]

* **首次返回时间的分布（原文式20）**

$$
P(T=t)=
\frac{(\sqrt2\,\varepsilon)^{1-4h}}
{\Gamma(\frac12-h)}\,
t^{(4h-3)/2}
\exp\left(-\frac{2\varepsilon^2}{t}\right).
$$

其中 $\Gamma$ 为Gamma函数，$\varepsilon$ 为离开阈值时的微小初始偏移，$h$ 与上式相同。这是按论文印出的理论表达保留。阅读核查中发现，该式Gamma函数参数与时间幂指数的归一化一致性仍需核对；这里保留原式，不自行修正归一化常数。

在指数截断因子影响较小的时间范围，原文据此写出：

$$
P(T)\sim T^{-\alpha},\qquad \alpha=\frac32-2h.
$$

这说明在该理论图景中，平均驱动偏置可以影响返回时间指数。作者同时说明：**宏观返回过程的指数怎样与实际spike雪崩指数严格对应，仍需要进一步研究。**[^limits]

# 3.通过网络模型进行的实验以及结论

## 实验1：只改变抑制的速度，网络活动会怎样变化？

**实验目的：** 先弄清楚“E-I平衡”能否与群体同步共存。作者比较抑制跟得上兴奋，以及抑制慢几毫秒才跟上的两种情况。[^exp1]

**实验过程：**

Step 1：保持网络规模、连接概率、权重和外部输入不变，令 $Q_o=5$ Hz。E突触的衰减时间固定为2 ms。

Step 2：建立快抑制条件，令 $\tau_d^I=1$ ms；再建立慢抑制条件，令 $\tau_d^I=3.5$ ms。记录E/I细胞的spike raster、E群体放电率和单细胞收到的平均E/I电流。

Step 3：把E电流、I电流和二者的差画在一起，看兴奋出现后，抑制能多快抵消它；同时计算单细胞ISI的CV，检查群体振荡是否意味着每个细胞都变成周期放电。

**实验结果与结论：**

快抑制时，抑制反馈能很快抵消兴奋电流，群体放电率近乎恒定，raster表现为异步活动。作者称这种情况为 **strict balance**。

慢抑制时，兴奋电流会先占优势，在抑制追上来前留出一小段集体放电窗口；随后抑制占优势，活动下降，于是群体活动开始起伏。作者称这种情况为 **loose balance**：E与I仍在较长时间尺度上平衡，但快速时间尺度上可以轮流占优势。

两种条件下，单细胞ISI的CV都在1附近。也就是说，**群体出现振荡时，单个细胞仍可以不规则地放电**。这里的群体节律不要求所有细胞每一轮都发放。

**对应图：** Fig. 1A–I，重点看B/C的raster、E/F的群体放电率、G的CV，以及H/I的输入电流。图在PDF第5页，结果解释在第8–9页。[^exp1]

## 实验2：连续扫描抑制衰减时间，找到同步转变的过程

**实验目的：** 实验1只比较了两个例子；这一步检查从快抑制到慢抑制的变化是否连续，以及单细胞、群体指标是否一起变化。[^exp2]

**实验过程：**

Step 1：保持 $Q_o=5$ Hz与其他默认参数，逐渐增大 $\tau_d^I$。总体扫描范围为1–4.5 ms。

Step 2：对每个条件，计算E细胞之间的平均PCC、膜电位coherence和群体放电率CV，观察同步和群体波动何时明显增强。

Step 3：同时计算单细胞ISI的CV、单细胞FF与群体FF。再比较“单细胞平均放电率/群体振荡峰值频率”，估计一个细胞参与每轮群体振荡的稀疏程度。

**实验结果与结论：**

随着抑制衰减变慢，网络从异步状态进入群体振荡。同步指标、群体放电率CV和群体FF都在转变附近明显上升；单细胞CV和FF却没有随群体同步一起大幅上升。

在振荡刚出现的区域，单细胞平均放电率约为群体振荡频率的1/10。按作者解释，每轮振荡大约只有10%的细胞参与，一个细胞也只随机参与少量振荡，因此能够保持较高的ISI不规则性。这就是文中反复提到的 **sparse synchrony，稀疏同步**。

当 $\tau_d^I$ 继续增大到约4 ms附近，网络出现更强的burst，参与比例才明显增大。因而不能把“刚出现振荡的稀疏同步”和“更慢抑制下的大范围burst”看成同一种活动强度。

这组实验把两个层次分开了：**群体波动增强，可以主要来自细胞之间的协调，而不要求每个细胞自身的放电规律发生同样程度的变化。**

**对应图：** Fig. 2D–G，PDF第9页；结果解释在第10页。[^exp2]

## 实验3：用平均场模型预测大网络的平均放电和群体波动

**实验目的：** 检查作者建立的宏观方程是否确实能对应微观LIF网络，而不只是定性画出一个会振荡的系统。这里可以把“平均场”理解为：不逐个追踪神经元，只追踪E/I群体平均膜电位与突触作用。[^exp3]

**实验过程：**

Step 1：先运行微观网络，在异步条件 $\tau_d^I=1$ ms下，改变外部Poisson频率 $Q_o$，记录平均放电率；同时改变网络规模进行比较。

Step 2：用微观网络异步状态下的平均膜电位和放电率，估计平均场所需的有效参数 $\sigma_E$、$\sigma_I$。因此这套方法有数值校准步骤，作者称其为**半解析平均场**。

Step 3：把宏观方程给出的平均放电率与微观网络结果对照。Fig. 2B的蓝色结果使用逐个输入条件从微观数据校准的有效参数；绿色结果另用外部白噪声近似估计有效参数，并固定在 $Q_o=5$ Hz对应的值。绿色方案使用的近似为：

$$
\sigma_\alpha=\sqrt{J_{\alpha o}^2n_oQ_o\tau_\alpha},\qquad Q_o=5\,\mathrm{Hz}.
$$

这里保持频率与膜时间常数的单位一致。它属于原文外部高斯白噪声近似的参数估计，与蓝色方案逐条件从微观数据校准不同。

Step 4：对群体放电率时间序列和群体FF也做对照。稳定性分析时去掉平均场噪声，数值模拟宏观波动时则保留噪声，以考虑有限规模网络的波动。

**实验结果与结论：**

在不同输入强度下分别校准有效参数，平均场对网络平均放电率的估计与微观模拟较一致。把有效参数固定后，精确放电率会出现偏差，但仍能抓住输入增强时平均放电率近似线性增加的趋势。

平均场也能表现快抑制下近乎平稳、慢抑制下振荡的群体活动，并定性反映群体FF的变化。因此作者得到一个可用来分析网络宏观状态的低维描述。

正文还报告了更换外部驱动形式的检查：把Poisson输入换成高斯白噪声或常量输入，宏观方程仍能近似网络活动。主文没有在这里给出全部对照参数和单独的主图，不能据此补写一套具体刺激协议。

这个结果的范围要记清楚：平均场主要解释群体统计；有效参数吸收了部分未显式处理的相关性、不应期等影响。它没有完整保留每个spike的细时间结构，后面的小雪崩不能直接从宏观曲线逐一还原。

Fig. 2B还检查了不同网络规模；其中5,000和7,500细胞条件下，权重分别乘1.41和1.15，以保持常用的随规模变化的权重标度，不能把它写成“只改细胞数、其他参数完全不变”。

**对应图：** Fig. 2A–B，PDF第9页；Fig. 1E–F，第5页；Fig. 2G，第9页。方法见第4–6页，结果说明见第9–10页。[^exp3]

## 实验4：平均场中的Hopf分岔能否定位同步转变？

**实验目的：** 给前面观察到的转变找一个动力学解释：为什么抑制只慢几毫秒，原先平稳的群体活动就开始振荡？这是理论分析与微观模拟的对照。[^exp4]

**实验过程：**

Step 1：对不含噪声的平均场方程求稳定状态。快抑制时，群体平均活动可以停在一个稳定固定点附近；加入小波动后，活动会回落到这个状态。

Step 2：逐渐增大 $\tau_d^I$，检查固定点附近的扰动是否仍会衰减。作者对第1.3节式15的雅可比矩阵求主导特征值，检查原先衰减的振荡扰动何时开始增长。

Step 3：把理论预测的转变位置、振荡频率与微观网络的同步指标、群体放电率功率谱对照。

**实验结果与结论：**

在主图参数下，转变发生在 $\tau_d^I\approx3$ ms附近。平均场的稳定固定点经过**超临界Hopf分岔**失稳，随后出现幅度从小到大增长的稳定周期运动。

直观理解就是：抑制跟得足够快时，活动一旦升高很快就会被压回去；抑制慢下来后，反馈形成有节律的增大与压低，于是群体平均活动开始持续起伏。

临界点附近，理论给出的振荡频率与微观网络测得的谱峰频率较接近，同步指标也在这个区域明显增强。这样，**“抑制时间尺度变化→固定点失稳→群体振荡出现”**就把参数扫描和动力学机制接了起来。

这里的“超临界Hopf分岔”是分岔类型名称；“超临界雪崩状态”则描述雪崩统计，两个词不要直接当作同一个分类。

**对应图：** Fig. 2C–E，PDF第9页；稳定性分析方法见第6页，结果解释见第10页。[^exp4]

## 实验5：同步转变附近是否真的出现临界雪崩？

**实验目的：** 前面找到的是振荡转变；这一步检查临界雪崩是否出现在同一个参数区域，并把“图像像幂律”和“通过统计检验”分开。[^exp5]

**实验过程：**

Step 1：沿 $\tau_d^I$ 扫描，用第2节的空bin方法提取E群体雪崩。先计算大小分布与拟合幂律的距离 $D$，找出最接近幂律的候选区域。

Step 2：重点比较三个条件：$\tau_d^I=1.2$ ms、3 ms、4.3 ms。分别观察raster、群体放电率PSD、雪崩大小分布、持续时间分布，以及给定持续时间下的平均大小。

Step 3：对候选临界区，使用NCC toolbox做最大似然估计和双端截断KS检验。小雪崩容易受离散统计影响，大雪崩容易受有限规模影响，所以作者只在满足要求的截断区间估计指数。

Step 4：从 $P(S)$、$P(T)$ 分别估计 $\tau$、$\alpha$，再在相同持续时间拟合范围内，用加权最小二乘拟合 $\langle S\rangle(T)$ 的指数，检查它与 $(\alpha-1)/(\tau-1)$ 是否接近。

**实验结果与结论：**

在1.2 ms条件下，活动主要异步，PSD没有明显集体振荡峰，大小与持续时间分布接近指数衰减，作者将其归为**亚临界**。

在3 ms附近，$D$ 最小，大小与持续时间分布在截断区间内通过幂律检验，指数之间近似满足标度关系；同时网络刚出现快速集体振荡，细胞间平均相关性仍较弱。作者据此把这里识别为**临界区域**。

在4.3 ms条件下，群体振荡和大范围burst更明显；大小、持续时间分布相对临界条件出现更重的尾部，特定尺度的大事件增多，作者将其归为**超临界**。

因此，论文的主要发现是：**临界雪崩出现在异步活动向稀疏同步转变的附近；抑制更慢、群体振荡更强，并不意味着更加临界。**

Fig. 3G也提醒：平均大小与持续时间之间出现幂律关系，在亚临界条件下也可能看到，单独这一项仍不能完成临界性判断。本文采用的是分布检验加指数关系的联合证据。

图中输入条件也要分清：Fig. 3A–D用 $Q_o=5$ Hz；Fig. 3E–G展示的例子用8 Hz。比较曲线时需要同时核对衰减时间和输入频率。

**对应图：** Fig. 3A–G，PDF第11页；拟合方法见第7–8页，结果解释见第10–12页。[^exp5]

## 实验6：保持临界附近，改变背景输入会不会改变临界指数？

**实验目的：** 检查临界性是否必须对应一组固定的指数，也检查网络接收的背景驱动能否改变雪崩统计。[^exp6]

**实验过程：**

Step 1：把抑制衰减时间固定在转变附近 $\tau_d^I=3$ ms，保持网络结构和权重不变。

Step 2：把每路外部Poisson输入的频率 $Q_o$ 从4 Hz改变到8 Hz。每个输入条件重新提取雪崩；由于箱宽取合并脉冲序列的平均ISI，不能默认不同输入条件下箱宽相同。

Step 3：重复大小和持续时间分布的统计检验，得到各条件的 $\tau$、$\alpha$ 和平均大小—持续时间指数；把直接拟合指数与分布指数预测值画在一起。

**实验结果与结论：**

这一区间内，网络仍显示显著的临界雪崩性质，但指数会随背景输入频率变化。尽管指数不同，三个指数之间的标度关系仍近似保持。

因此本文并不把 $\tau=1.5$、$\alpha=2$ 当作必须满足的固定标准。改变驱动后，“分布指数变了”和“系统离开临界状态”不能直接画等号，需要继续检查分布与标度关系。

作者还观察到 $\tau$ 与 $\alpha$ 近似线性相关，并把“轻微调整背景输入，可以在保持临界统计的同时改变指数”作为可供后续实验检验的预测。这里验证的是模型，不代表已经在真实切片上做了同样的外部输入操控。

**对应图：** Fig. 3H及插图，PDF第11页；结果与预测的说明见第12页。[^exp6]

# 4.文中使用的真实实验数据

## 数据来源

作者再分析了Ito等人公开的小鼠体感皮层器官型切片培养数据，来源为CRCNS的 **ssc-3** 数据集。共有25组记录，通常每组1 h，Set 19为48 min；切片培养2–4周后，使用512电极阵列记录无刺激条件下的自发spike，并用PCA相关方法分选。**这部分是公开数据再分析，本文没有新做一套动物记录实验。**[^data]

## 实验7：真实切片的up-state中，能否同时看到不规则放电和临界雪崩？

**实验目的：** 模型在均匀随机连接下得到的共存现象，能否在真实神经活动中找到相似证据？作者只比较数据中与模型持续活跃状态较接近的up-state。[^exp7]

**实验过程：**

Step 1：先区分up/down-state。群体放电率按10 ms分箱，再用100 ms方窗平滑；高于该数据集最大群体放电率30%、并持续超过1 s的时段定义为up-state。按原文文字写为：

$$
q_{\rm smooth}(t)>0.3\max_t q_{\rm smooth}(t),\qquad T_{\rm up}>1\,\mathrm{s}.
$$

其中 $q_{\rm smooth}$ 是平滑后的群体放电率，$T_{\rm up}$ 为该活跃时段的持续时间。

Step 2：比较up/down-state的单细胞放电率分布和群体PSD。数据中两种状态约以0.1 Hz缓慢交替，up-state里活动更充分、群体振荡也更明显，因此作者主要在这些时段继续分析。

Step 3：在up-state内计算单细胞ISI的CV。然后用这些时段合并spike train的平均ISI作为雪崩箱宽，按连续非空bin划分雪崩；不把30%的up-state筛选阈值继续当作雪崩边界。

Step 4：对25组数据逐一检验大小与持续时间是否同时满足幂律拟合要求。对于两项都通过的数据，进一步比较直接拟合的平均大小—持续时间指数与分布指数预测值。

Step 5：做随机化对照。在每个up-state内，随机打乱各细胞的ISI顺序，破坏原来的时间相关结构，再重新计算雪崩分布，与原始数据比较。

**实验结果与结论：**

25组数据中，**9组的大小和持续时间分布都通过作者的幂律判据；其中7组进一步近似满足标度关系，误差小于0.1**。Fig. 4D、H展示的重点是这7组，不能概括为25组全部都临界。

这些数据中，单细胞ISI的CV典型值约1–1.5，表明放电仍然不规则；同一批数据同时具有临界雪崩统计。Set 1给出了大小、持续时间分布与平均大小—持续时间关系的具体例子。

打乱ISI后，原来的幂律分布不能保持，说明结果依赖数据自身的时间组织。这个对照支持“时间相关结构参与了雪崩统计”，但不能单凭它确认真实切片也一定由同一Hopf分岔机制产生临界性。

真实数据的指数在不同记录间也有变化；它们的标度关系和模型接近，但指数之间线性关系的斜率并不完全一致。作者据此支持的是**不规则单细胞放电与临界群体活动能够共存**，没有声称模型复现了真实数据的全部性质。

**对应图：** Fig. 4A–H，PDF第13页；筛选与统计方法见第8页，结果说明见第13–14页。[^exp7]

模型在这里支持的是两种尺度现象的共存；它没有复现真实数据缓慢的up/down-state切换及完整的细胞放电率异质性。正文把形状塌缩列为后续更严格的检验方向，并未把它作为已完成的主要验证。[^limits]

**与前两篇笔记对照：** 本篇的spike雪崩采用连续非空bin定义；[[Cortical Circuit Dynamics Are Homeostatically Tuned to Criticality In Vivo|Ma2019]]则先把每个细胞每bin的活动记为0/1，再按群体活动百分位阈值划分事件。本篇权重固定，通过扫描抑制时间尺度定位临界区；[[Synaptic Plasticity Enables Adaptive SelfTuning Critical Networks|Stepp2015]]研究的是依赖可塑性的自调节，两者的实验问题不同。

## 论文来源

[论文页面](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2020.580011/full) · [本地 PDF](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf)

原文为2020年正式出版版，共19页。文中页码均按这份本地PDF的实际页序标注。

[^model]: [本地原文第3页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=3)；原文PDF第3–4页，Materials and Methods → Spiking Neuronal Network，含默认规模、连接、输入、LIF参数、固定权重、积分方法及模拟次数。
[^metrics]: [本地原文第6页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=6)；原文PDF第6–7页，Statistical Analysis of Model and Experiment Data → Spike Count Series；Quantifying Spiking Irregularity and Firing Rate Variability；Quantifying Network Synchrony。
[^avalanche]: [本地原文第7页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=7)；原文PDF第7页，Neuronal Avalanche Analysis；Fig. 3B的雪崩示例见完整PDF第11页。
[^statistics]: [本地原文第7页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=7)；原文PDF第7–8页，Neuronal Avalanche Analysis，含距离D与NCC双端截断检验；三个指数关系见完整PDF第12页式(16)。
[^exp1]: [本地原文第5页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=5)；原文PDF第5页完整Fig. 1及图注；第8–9页完整Results → Network Synchrony Arises From Loose Temporal Balance。
[^exp2]: [本地原文第9页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=9)；原文PDF第9页完整Fig. 2D–G及图注；第10页解释PCC、coherence、CV、FF和稀疏参与比例。
[^exp3]: [本地原文第4页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=4)；原文PDF第4–6页平均场方法；第5页完整Fig. 1E–F；第9页完整Fig. 2A–B、G及图注；第9–10页说明有效参数校准、固定参数比较与网络规模对照。
[^exp4]: [本地原文第6页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=6)；原文PDF第6页Analysis of the Steady-State Dynamics；第9页完整Fig. 2C–E及图注；第10页解释超临界Hopf分岔与频率预测。
[^exp5]: [本地原文第7页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=7)；原文PDF第7–8页雪崩统计方法；第11页完整Fig. 3A–G及图注；第10–12页Scale-Free Neuronal Avalanches Near the Critical Transition Point。
[^exp6]: [本地原文第11页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=11)；原文PDF第11页完整Fig. 3H及插图、图注；第12页说明4–8 Hz输入扫描、指数变化、标度关系及待实验检验的预测。
[^data]: [本地原文第8页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=8)；原文PDF第8页Experimental Data Analysis；第16页Data Availability Statement明确公开数据地址为 http://crcns.org/data-sets/ssc/ssc-3；第17页列Ito等人2014、2016的来源。
[^exp7]: [本地原文第8页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=8)；原文PDF第8页完整Experimental Data Analysis；第13页完整Fig. 4及图注、Coexistence of Irregular Spiking and Critical Avalanches in Experimental Data；第14页明确25→9→7组的筛选结果、CV及打乱ISI对照。
[^limits]: [本地原文第12页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=12)；原文PDF第12–13页宏观首返解释与微观/宏观对应的限制；第15页讨论形状塌缩及平均场近似；第16页Outlook说明异质性、适应机制和up/down-state限制。
[^meanfield]: [本地原文第4页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=4)；原文PDF第4–6页Mean-Field Theory of Network Dynamics，式3–12。式7–10使用正文明确的1 ms计数约定，式9、10含π/√3因子。
[^stability]: [本地原文第6页](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2020_Frontiers%20in%20Systems%20Neuroscience_Hopf%20Bifurcation%20in%20Mean%20Field%20Explains%20Critical%20Avalanches%20in%20Excitation-Inhibition%20Balanced%20Neuronal%20Networks%20-%20A%20Mechanism%20for%20Multiscale%20Variability.pdf#page=6)；原文PDF第6页Analysis of the Steady-State Dynamics，式13–15；第10页解释Hopf分岔、共轭特征值越过虚轴与频率ω/(2π)。
