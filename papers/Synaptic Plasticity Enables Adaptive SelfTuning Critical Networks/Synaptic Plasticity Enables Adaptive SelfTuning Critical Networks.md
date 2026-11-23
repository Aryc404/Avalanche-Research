# 文中使用的模型

LIF 神经元模型

网络中一共包含10000个神经元
其中 兴奋性神经元(E) 有 8000个，抑制性神经元(I) 有 2000个
神经元之间并非全部互相连接，初始连接随机生成，神经元间的各类型的连接概率为 **1%**
包含E→E,E→I,I→E,I→I 四类突触
所有已建立的突触初始的电导均为0.5nS
* 论文中主要通过改变电导来改变连接的强弱
模拟时间共900s，time step 为1ms 因此总共运行了900000个time steps

论文中使用的核心公式:

* **膜电位的变化方程**
	${}\begin{aligned} \tau_m\frac{dV_i}{dt} ={}&(V_{\mathrm{rest}}-V_i)\\ &+(E_{\mathrm{ex}}-V_i) \sum_{j\in Pre_E}u_{ij}x_{ij}g_{w,ij}\\ &+(E_{\mathrm{inh}}-V_i) \sum_{j\in Pre_I}u_{ij}x_{ij}g_{z,ij}. \end{aligned}{}$
	其中：
		${}V_{i}{}$  第 i 个 神经元的膜电位
		${}\tau_{m}{}$  膜时间常数 文中取20ms
		${}V_{rest}{}$ **静息下的膜电位**（没有输入时 膜电位长期靠近的值) 文中取 -74mV
		${}E_{ex}{}$ 兴奋性反转电位 文中取 0mV
		${}E_{inh}{}$ 抑制性反转电位 文中取 -80mV
		${}Pre_{exc}{}$  连接到 神经元 i 的兴奋性突触前神经元集合。
		${}Pre_{inh}{}$  连接到 神经元 i 的抑制性突触前神经元集合。
		${}g_{w,ij}{}$ 神经元 j → i 的兴奋性突触当前的瞬时电导
		${}g_{z,ij}{}$  神经元 j → i 的抑制性突触当前的瞬时电导
		${}x_{ij}{}$ 当前可以利用的突触资源比例
		${}u_{ij}{}$ 资源利用率
		${}u_{ij}x_{ij}{}$ STP产生的短时突触效能
		
	当神经元的膜电位超过 放电阈值${}V_{T}{}$（文中取值为 -54mV)时神经元放电并且膜电位会重置为${}V_{reset}{}$（文中取值为 -60mV)
		论文中没有让神经元放电后膜电位回到${}Vrest{}$=-74mV是为了让神经元放完电以后离阈值 -54mV 更近，促进神经元在短时间内更容易再次放电

* **AMPA / GABA 突触电导变化公式**
	$$
	\frac{dg_w}{dt} = -\frac{g_w}{\tau_{\rm AMPA}} + w\sum_{s\in S_{\rm exc}}\delta(t-s)
	$$$$
	\frac{dg_z}{dt} = -\frac{g_z}{\tau_{\rm GABA}} + z\sum_{s\in S_{\rm inh}}\delta(t-s)
	$$
	含义:当神经元接收到 presynaptic spike时 $g_{w}$上升 之后 按照指数形式慢慢衰减
	
	其中：
		$g_{w}$ 为当前兴奋性突触的电导
		$g_{z}$ 为当前抑制性突触的电导
		$w$ 为一次兴奋性 spike 引起的电导增量，由 E-STDP 改变
		$z$ 为一次抑制性 spike 引起的电导增量，由 I-STDP 改变
		$\tau_{AMPA}$ 为AMPA 电导衰减时间常数 文中取19ms
		$\tau_{GABA}$ 为GABA 电导衰减时间常数 文中取14ms
		$S_{\rm exc},S_{\rm inh}$ 为兴奋、抑制突触前 spike 的时间集合
		$s$ 为某一个spike 的发生时间

* **E-STDP**
	$$
	w_{\rm new}=w_{\rm old}+\Delta w
	$$
	新的兴奋性突触强度等于原来的强度加上 STDP 修改量。

	$$
	\Delta w=g_{\max}^{\rm exc}F(\Delta t)
	$$

	$$
	F(\Delta t)=
	\begin{cases}
	A_+e^{\Delta t/\tau_+},&\Delta t<0\\
	-A_-e^{-\Delta t/\tau_-},&\Delta t\geq0
	\end{cases}
	$$
	其中：
		$\Delta t=t_{\rm pre}-t_{\rm post}$
		$w_{\rm old},w_{\rm new}$ 为更新前后的兴奋性突触强度
		$\Delta w$ 为一次 STDP 更新量
		$g_{\max}^{\rm exc}$ 为兴奋性突触允许的最大电导，文中取 $1.94\,\mathrm{nS}$
		$t_{\rm pre}$ 为突触前 spike 时间
		$t_{\rm post}$ 为突触后 spike 时间
		$\Delta t$ 为两者时间差
		$A_+$ 为 potentiation 的幅度参数
		$A_-$ 为 depression 的幅度参数
		$\tau_+,\tau_-$ 为 STDP 时间窗口，文中均取 $20\,\mathrm{ms}$

	如果 pre 先放电，$t_{\rm pre}<t_{\rm post}$，则 $\Delta t<0$、$\Delta w>0$，突触增强。反过来，$\Delta t>0$ 时，$\Delta w<0$，突触减弱。

	$$
	\beta=\frac{|A_-\tau_-|}{|A_+\tau_+|}
	$$
	$\beta$ 表示 depression 区域与 potentiation 区域的相对面积，文中取 $\beta=1.21$。

	另外${}w{}$被严格控制在小于$g_{\max}^{\rm exc}$的范围内

* **I-STDP**
	$$
	z_{\rm new}=z_{\rm old}+\Delta z
	$$

	$$
	\Delta z=
	\begin{cases}
	B_+e^{-|\Delta t|/\tau},&|\Delta t|\leq\tau\\[3pt]
	-B_-e^{-|\Delta t|/\tau},&|\Delta t|>\tau
	\end{cases}
	$$
	其中：
		$z$ 为抑制性突触强度
		$\Delta z$ 为 I-STDP 修改量
		$B_+=0.0015\,\mathrm{nS}$ 为增强幅度
		$B_-=0.0003\,\mathrm{nS}$ 为减弱幅度
		$\tau=10\,\mathrm{ms}$ 为 I-STDP 时间尺度
		$|\Delta t|$ 为 pre/post spike 的时间间隔绝对值
		$g_{\max}^{\rm inh}=4.74\,\mathrm{nS}$ 为最大抑制突触电导
		也即${}z{}$被严格控制在小于$g_{\max}^{\rm inh}$的范围内

	E-STDP 看 spike 顺序，而 I-STDP 更强调 pre、post spike 离得有多近，所以它的函数关于 $|\Delta t|$ 对称。

* **STP**
	$$
	\frac{dx}{dt}=\frac{1-x}{\tau_D}-ux\delta(t-t_{\rm sp})
	$$

	$$
	\frac{du}{dt}=\frac{U-u}{\tau_F}+U(1-u)\delta(t-t_{\rm sp})
	$$
	其中：
		$x$ 为剩余可用突触资源
		$u$ 为一次 spike 会调用多少比例的资源
		$ux$ 为当前实际突触效能
		$U=0.5$ 为 $u$ 的 baseline 水平
		$t_{\rm sp}$ 为 presynaptic spike 发生时间
		$\tau_D=26\,\mathrm{ms}$ 为资源恢复时间常数
		$\tau_F=41\,\mathrm{ms}$ 为 facilitation 的恢复时间常数

**STDP 改变的是长期突触强度 $w$ 或 $z$**
**而 STP 改变的是短时间内单次 spike 的实际作用大小 $ux$。**

# 测量网络稳态所用指标

* **Branching ratio**
	论文定义 spike train：
	$$
	X_i(n)=
	\begin{cases}
	1,&i\text{ 在时间步 }n\text{ firing}\\
	0,&\text{否则}
	\end{cases}
	$$
	并定义：
	$$
	G_{ij}=
	\begin{cases}
	1,&i\rightarrow j\text{ 有连接}\\
	0,&\text{没有}
	\end{cases}
	$$

	论文先计算某个神经元附近的 local branching ratio：
	$$
	s_i(n)=
	\frac{\text{该 spike 后面引发的 postsynaptic spikes}}
	{\text{该 spike 前面的 presynaptic spikes}}
	$$

	再对当前 firing 的 neurons 平均：
	$$
	s(n)=
	\frac{\sum_iX_i(n)s_i(n)}
	{\sum_iX_i(n)}
	$$

	论文主要使用 network-level branching ratio，把整个网络所有相关的 pre/post spike 一起统计：
	$$
	s(n)\approx\frac{\text{post activity}}{\text{pre activity}}
	$$
	论文使用 $f=1$、$D=3$。

	其中：
		$n$ 为当前时间步
		$D$ 为前后因果时间窗宽度
		$f$ 为时间窗与当前 spike 的偏移量
		$X_i(n)$ 表示第 $i$ 个神经元是否在 $n$ firing
		$G_{ij}$ 为连接矩阵
		$s_i(n)$ 为局部 branching ratio
		$s(n)$ 为整个网络 branching ratio

	$$s<1:\text{subcritical tendency}$$
	$$s>1:\text{supercritical tendency}$$
	$$s\approx1:\text{critical branching}$$

	$$
	\boxed{s\approx1\text{ 不能单独证明 criticality}}
	$$

* **Avalanche**
文中是怎么定义avalanche的？
	作者把第 $k$ 个 avalanche 写作：
	$$
	A_k
	$$
	论文把 avalanche 定义为一串具有因果连续性的 spikes。对于时间步 $n$ 的一个 spike，作者向前检查时间窗：
	$$
	[n-f-D,\;n-f-1]
	$$
	中是否存在与当前神经元有突触连接的 presynaptic spike。论文取：
	$$
	f=1,\qquad D=3
	$$
如果找到了这样的前驱 spike，而且它已经属于某个 avalanche，那么当前 spike 也加入同一个 avalanche；
如果没有找到，则当前 spike 作为一个新的 avalanche 起点。

一个 spike 可以同时属于多个 avalanche，因为作者允许多个可能的 presynaptic causes。

Avalanche size：
	$$
	S_k=|A_k|
	$$
	即这个 avalanche 包含的 spike 数量。

然后拟合：
	$$
	P(S)\sim S^\lambda
	$$
	临界状态理论期望：
	$$
	\lambda\approx-\frac32
	$$

* **DFA（Detrended Fluctuation Analysis，去趋势波动分析）**
	DFA 研究 firing 是否存在长程时间相关性。

	$$
	\overline Y_i=
	\frac1{N_i}\sum_{s=1}^{N_i}Y_i(s)
	$$
	其中：
		$Y_i(s)$ 为第 $i$ 个 neuron 的第 $s$ 个 ISI（inter-spike interval，脉冲间隔）
		$N_i$ 为总 ISI 数量
		$\overline Y_i$ 为平均 ISI

	将去均值的 ISI 累积起来：
	$$
	y_i(s)=
	\sum\left[Y_i(s)-\overline Y_i\right]
	$$

	把序列切成长度为 $l$ 的片段，在每个片段去除局部趋势：
	$$
	F^2(l)=
	\frac1M\sum_{m=1}^{M}(z_m-\widehat z_m)^2
	$$
	其中：
		$M$ 为片段数量
		$l$ 为每个片段长度
		$z_m$ 为原始第 $m$ 段
		$\widehat z_m$ 为该段拟合出来的趋势（最小二乘法拟合直线)

	最后看：
	$$
	F^2(l)\propto l^\alpha
	$$
	如果 $0.5<\alpha<1$，说明存在正的 long-range temporal correlations。


总结一下 文中把稳态网络的要求列为三项:

**1.branching ratio 接近 1**

**2.avalanche exponent 接近 ${}-\frac{3}{2}{}$**

**3.DFA 指数  $0.5<\alpha<1$**

# 文中所做实验 与 结论
**实验1：**
  
实验过程：
	 作者构建 10,000 个 LIF 神经元的 E-I 网络。模拟开始时，只对随机选取的 20 个兴奋性神经元施加 300 Hz 的 Poisson spike，持续 15 ms；之后完全撤掉外部驱动，让网络依靠自身循环连接和突触可塑性运行 900 s。 

实验结论：
	网络在短暂初始化后能够维持 PABA（**persistent asynchronous background activity**）：
		持续的(外部刺激停止以后，网络自己还能继续放电)
		异步的(不同神经元不是整齐同步一起 firing，而是各自不规则地发放)
	平均firing rate  $\approx31.8\pm2.6\text{Hz}$
	spike 之间相关性很低，活动呈异步、不规则状态。
	网络表现出上述三个临界性特征

**实验2：**

实验过程：
	作者建立两个完全相同的网络。
	其中一个正常运行；另一个在某一时刻人为让一个本来不会 firing 的神经元多产生一个 spike，然后比较两者之后的 spike pattern。

实验结论:
	仅仅增加一个 spike，就会导致两个网络的 spike state 很快开始分离。
	这说明网络对非常小的扰动 有着高度的敏感

**实验3：**

实验过程:
	作者对一部分兴奋性神经元施加强刺激。
	一次 perturbation 包含10个pulses
	每个 pulse 为 300Hz 持续500ms
	这样的 perturbation 分别在 300s 400s 500s 施加

实验结论:
	扰动期间，网络明显偏离原来的临界状态，大的avalanche 增多。
	但扰动结束后 网络又回到了临界状态 且再受扰动后能自行调节回去

**实验4：**

实验过程:
	作者比较：
	没有受到 perturbation 的网络
	受到三次强 perturbation 的网络
	然后观察兴奋性、抑制性突触权重随时间的变化，并计算两个网络突触权重之间的 **MSE**
		MSE: Mean Squared Error 用于衡量两组突触权重之间的差异
		$$
		MSE= \frac{\|w_1-w_2\|^2}{N}
		$$
		其中：
			$w_1$ 为一个网络所有突触权重组成的向量
			$w_2$ 为另一个网络对应的权重向量
			$N$ 为突触总数量
			MSE 为两个网络突触结构差异的平均平方量

实验结论:
	每次 perturbation 都会让 E 和 I 突触权重发生明显变化。
	特别是：
	兴奋性权重变化具有较强的长期保留；
	抑制性权重在扰动后更容易恢复。
	因此虽然网络最终重新回到临界状态，**它回到的并不是原来完全相同的网络**
	换句话说：
	**临界动力学可以恢复，但网络连接结构已经被学习改变**


**实验5：**

实验过程:
	作者让网络先正常运行，然后在 t = 300 s 时 关闭inhibitory STDP
	再关闭 inhibitory STDP，也即固定 z，其他机制继续运行。

实验结论:
	在关闭 I STDP时，avalanche size distribution 逐渐偏离 ${}-\frac{3}{2}{}$
	临界性丢失
	但是分支比 $s$ ≈ 1  同时E I balance 也仍然接近稳定
	这代表着:$$\boxed{\text{E/I balance}\neq\text{criticality}}$$
	
	