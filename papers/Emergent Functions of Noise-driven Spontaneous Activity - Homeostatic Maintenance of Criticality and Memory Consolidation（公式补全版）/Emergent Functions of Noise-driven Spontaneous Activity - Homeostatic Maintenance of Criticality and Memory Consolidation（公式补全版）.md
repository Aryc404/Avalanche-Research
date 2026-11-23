# 1.文中使用的网络模型

**Conductance-based LIF with escape noise：带逃逸噪声的电导型泄漏积分发放模型。**

模型一共包含 **100个神经元：80个兴奋性神经元E、20个抑制性神经元I**。

网络为fully connected，包含E→E、E→I、I→E、I→I四类突触；**所有突触权重初始为0**。刚开始没有有效的循环突触驱动，神经元先依靠噪声随机放电，之后STDP逐渐形成非零连接强度。“全连接”描述允许存在的连接，不等于一开始所有连接都很强。[^model]

网络中每个神经元有膜电位$v$、兴奋性电导$g_{\mathrm{exc}}$、抑制性电导$g_{\mathrm{inh}}$和可用资源$x$四个状态变量；每条突触有自己的长期权重$w$。数值模拟使用Brian 2，time step为$\Delta t=0.1\,\mathrm{ms}$；每个参数条件运行30个不同初始随机种子。

**输入主要有两部分：**

- 自发活动阶段：escape noise产生随机spike，神经元之间再通过循环突触相互驱动。静息噪声放电参数为$f_{\mathrm{rest}}=0.4\,\mathrm{Hz}$。
- 重复刺激阶段：每种模式选6个E神经元，将这些神经元的膜电位同时提高$30\,\mathrm{mV}$；共4种空间模式。详细训练与测试流程见后文实验。

论文中使用的核心公式如下。式号对应用户提供的修正版PDF；其余为了说明含义写出的等价式，会单独注明。

## 1.1 膜电位的变化方程

**原文式(1)，PDF第2页：**

$$
\tau_m\frac{dv}{dt}
=(v_{\mathrm{rest}}-v)g_{\mathrm{rest}}
+(E_{\mathrm{exc}}-v)g_{\mathrm{exc}}
+(E_{\mathrm{inh}}-v)g_{\mathrm{inh}}.
\tag{1}
$$

其中：

- $v$：当前神经元的膜电位。
- $\tau_m=30\,\mathrm{ms}$：膜时间常数。
- $v_{\mathrm{rest}}=-74\,\mathrm{mV}$：静息膜电位。
- $g_{\mathrm{rest}}=1$：用于归一化的静息电导；原文说明电导已经按$g_{\mathrm{rest}}$归一化。
- $E_{\mathrm{exc}}=0\,\mathrm{mV}$：兴奋性反转电位。
- $E_{\mathrm{inh}}=-80\,\mathrm{mV}$：抑制性反转电位。
- $g_{\mathrm{exc}}$：当前神经元收到的全部兴奋性突触输入的总电导。
- $g_{\mathrm{inh}}$：当前神经元收到的全部抑制性突触输入的总电导。

**对公式的理解：**

第一项让膜电位向$-74\,\mathrm{mV}$回落。第二、三项分别把膜电位向兴奋性和抑制性反转电位拉动；突触电流取决于电导，也取决于当前的电位差。因此，同样的电导输入作用于不同膜电位时，产生的电流可以不同。[^model]

## 1.2 AMPA／GABA突触电导的变化

**没有新spike到达时，原文式(2)、(3)，PDF第2页：**

$$
\tau_{\mathrm{AMPA}}\frac{dg_{\mathrm{exc}}}{dt}
=-g_{\mathrm{exc}}.
\tag{2}
$$

$$
\tau_{\mathrm{GABA}}\frac{dg_{\mathrm{inh}}}{dt}
=-g_{\mathrm{inh}}.
\tag{3}
$$

其中，$\tau_{\mathrm{AMPA}}=2\,\mathrm{ms}$、$\tau_{\mathrm{GABA}}=4\,\mathrm{ms}$，分别是兴奋性和抑制性电导的衰减时间常数。

由这两个微分方程可以得到指数衰减形式（等价改写）：

$$
g_{\mathrm{exc}}(t)=g_{\mathrm{exc}}(t_0)
e^{-(t-t_0)/\tau_{\mathrm{AMPA}}},
\qquad
g_{\mathrm{inh}}(t)=g_{\mathrm{inh}}(t_0)
e^{-(t-t_0)/\tau_{\mathrm{GABA}}}.
$$

**有突触前spike到达时，原文式(5)、(6)，PDF第3页：**

$$
g_{\mathrm{exc}}\leftarrow
g_{\mathrm{exc}}+Uxw g_{\mathrm{exc}}^{\max}.
\tag{5}
$$

$$
g_{\mathrm{inh}}\leftarrow
g_{\mathrm{inh}}+Uxw g_{\mathrm{inh}}^{\max}.
\tag{6}
$$

其中：

- $\leftarrow$：事件发生时，变量瞬间按右边的结果更新。
- $U=0.4$：一次spike使用可用资源的比例。
- $x$：该突触前神经元目前可以使用的资源比例。
- $w$：该条突触的长期权重，限制在$[0,1]$。
- $g_{\mathrm{exc}}^{\max}=4.0$、$g_{\mathrm{inh}}^{\max}=4.0$：将权重和资源利用量换算成电导增量的比例参数。原表没有标注nS；这里沿用归一化电导，不补加物理单位。

E→E连接的传输延迟为$1.5\,\mathrm{ms}$，其他连接为$0.8\,\mathrm{ms}$。spike发出后，经过对应延迟才使突触后神经元的电导增加。

**理解：** 一次spike的实际作用大小是$Uxw g^{\max}$。即使长期权重$w$没有改变，可用资源$x$减少，也会让下一次spike的作用减弱。这里$g^{\max}$是电导增量的比例参数；多次输入可以叠加，不能将它直接理解为总瞬时电导$g$永远不能超过的上限。[^model]

## 1.3 Escape noise：神经元怎样决定是否放电

**原文第3页先定义瞬时发放强度：**

$$
\rho(t)=\frac{1}{\tau}
\exp\!\left(\frac{v(t)-v_{\mathrm{th}}}{b}\right).
$$

其中：

- $\rho(t)$：瞬时发放强度，亦可理解为hazard rate，量纲为$1/\mathrm{time}$；它本身不是一个时间步内的无量纲概率。
- $\tau$：控制噪声发放强度的时间尺度，**与膜时间常数$\tau_m$不同**。
- $v_{\mathrm{th}}=-54\,\mathrm{mV}$：指数发放曲线的位置参数，原文称firing threshold。
- $b=4\,\mathrm{mV}$：电位变化对发放强度的影响尺度。

设$S(t)$为从0到$t$一直没有出现spike的生存概率，原文给出：

$$
\frac{dS(t)}{dt}=-\rho(t)S(t),
\qquad
S(t)=\exp\!\left[-\int_0^t\rho(t')\,dt'\right].
$$

因此，一个时间步内出现spike的概率为：

$$
P_{\mathrm{spike}}(t,t+\Delta t)
=1-\exp\!\left[-\int_t^{t+\Delta t}\rho(t')\,dt'\right].
$$

其中$t'$是积分变量，$\Delta t$是数值模拟步长。若这个时间步内的$\rho$近似不变，上式为$1-e^{-\rho(t)\Delta t}$。进一步使用小时间步的一阶近似，得到**实际数值模拟使用的原文式(4)：**

$$
P_{\mathrm{spike}}(t,t+\Delta t)
\simeq
\min\!\left[
\frac{\Delta t}{\tau}
\exp\!\left(\frac{v(t)-v_{\mathrm{th}}}{b}\right),\ 1
\right].
\tag{4}
$$

**这里需要区分：** 积分式是从发放强度得到概率的表达；式(4)是原文采用的一阶近似，并用$\min(\cdot,1)$防止数值大于1。

原文还用静息状态的噪声放电率$f_{\mathrm{rest}}$设定$\tau$：

$$
\frac{\Delta t}{\tau}
=f_{\mathrm{rest}}\Delta t
\exp\!\left[-\frac{v_{\mathrm{rest}}-v_{\mathrm{th}}}{b}\right].
$$

其中$f_{\mathrm{rest}}=0.4\,\mathrm{Hz}$。代入参数可得$\tau\approx16.8\,\mathrm{ms}$，$\Delta t/\tau\approx0.00594$。当$v=v_{\mathrm{rest}}$时，单个$0.1\,\mathrm{ms}$时间步的发放概率约为$0.4\times10^{-4}=4\times10^{-5}$。

**理解：** 膜电位越高，spike概率越大；低于$v_{\mathrm{th}}$时也可能放电，达到$v_{\mathrm{th}}$时也不是立即必然放电。因此，本模型应称为带escape noise的LIF，不能按确定性硬阈值LIF解释。

放电后，膜电位重置为$v_{\mathrm{rest}}=-74\,\mathrm{mV}$，并进入不应期。Table 1中不应期两行的名称都印为$T_{\mathrm{exc}}^{\mathrm{ref}}$，数值分别为3 ms和2 ms，存在标签重复；笔记保留这一原文歧义，不直接替作者修正第二行的细胞类型。[^model]

## 1.4 STP：资源耗竭与恢复

**spike之间，原文式(7)，PDF第3页：**

$$
\tau_{\mathrm{rec}}\frac{dx}{dt}=1-x.
\tag{7}
$$

**每次突触前spike发生时，原文式(8)：**

$$
x\leftarrow x-Ux=(1-U)x.
\tag{8}
$$

其中：

- $x$：突触前神经元可用的资源比例。
- $U=0.4$：每次spike消耗当前资源的40%。
- $\tau_{\mathrm{rec}}=150\,\mathrm{ms}$：资源向1恢复的时间常数。

**理解：** 连续spike间隔很短时，$x$来不及恢复，下一次电导增量$Uxw g^{\max}$就更小；较长时间没有spike时，$x$逐渐回到1。

这里没有另一条动态facilitation变量$u$的更新方程，$U$是常数。因此，不能把[[Synaptic Plasticity Enables Adaptive SelfTuning Critical Networks|Stepp2015笔记]]中的$x$、$u$双变量STP和另一组参数直接搬到本文。

**STDP改变长期权重$w$；STP通过$x$改变近期spike的实际效能。**[^plasticity]

## 1.5 STDP的事件更新与权重范围

原文把pre和post两种spike事件分开写，**时间差统一为$t_{\mathrm{post}}-t_{\mathrm{pre}}$**。这一符号方向与Stepp2015笔记中采用的$t_{\mathrm{pre}}-t_{\mathrm{post}}$相反，不能只看正负号判断增强或减弱。

**突触前神经元在$t_{\mathrm{pre}}$放电时，原文式(9)，PDF第4页：**

$$
w\leftarrow\operatorname{Clip}\!\left(
w+\sum_{t_{\mathrm{post}}\in T_{\mathrm{post}}(t_{\mathrm{pre}})}
F(t_{\mathrm{post}}-t_{\mathrm{pre}}),\ 0,\ 1
\right).
\tag{9}
$$

**权重裁剪函数，原文式(10)：**

$$
\operatorname{Clip}(w,a,b)=\min\!\bigl(\max(w,a),b\bigr).
\tag{10}
$$

**突触后神经元在$t_{\mathrm{post}}$放电时，原文式(11)：**

$$
w\leftarrow\operatorname{Clip}\!\left(
w+\sum_{t_{\mathrm{pre}}\in T_{\mathrm{pre}}(t_{\mathrm{post}})}
F(t_{\mathrm{post}}-t_{\mathrm{pre}}),\ 0,\ 1
\right).
\tag{11}
$$

其中：

- $w$：当前这条连接的长期突触权重。
- $F$：根据突触前神经元是E还是I，分别选用下文的$F_E$或$F_I$。
- $T_{\mathrm{post}}(t_{\mathrm{pre}})$：当前pre spike之前的post spike时间集合。
- $T_{\mathrm{pre}}(t_{\mathrm{post}})$：当前post spike之前的pre spike时间集合。原文说明文字有一处集合下标排印问题，这里按式(11)的含义书写。
- $a,b$：裁剪下限和上限；本文实际取0和1。这里裁剪函数的$b$是上限参数，与escape noise中$b=4\,\mathrm{mV}$的符号含义不同。

等价地，可以把一次更新概括成（说明用改写）：

$$
w_{\mathrm{new}}=\min\!\bigl(1,\max(0,w_{\mathrm{old}}+\Delta w)\bigr),
$$

其中$\Delta w$为该次事件按学习窗累加的总变化。无论是兴奋性还是抑制性连接，$w$都非负；突触作用的正负由式(1)里的反转电位决定，不需要把抑制性权重写成负数。[^plasticity]

## 1.6 E-STDP：不对称的兴奋性学习窗

**原文式(12)，PDF第4页：**

$$
F_E(\delta)=
\begin{cases}
A_E\exp\!\left(-\dfrac{\delta}{\tau_E}\right),
&\delta\geq0,\\[5pt]
-A_E\beta_E\exp\!\left(\dfrac{\delta}{\tau_E}\right),
&\delta<0.
\end{cases}
\qquad
\delta=t_{\mathrm{post}}-t_{\mathrm{pre}}.
\tag{12}
$$

原文以$t$表示学习窗的自变量；这里换成$\delta$，避免与网络运行时间$t$混淆，函数本身不变。

其中：

- $F_E(\delta)$：一对pre／post spike对兴奋性突触权重的改变。
- $A_E=0.02$：增强一侧的幅度。
- $\tau_E=20\,\mathrm{ms}$：学习窗的时间常数。
- $\beta_E$：减弱相对增强的强度参数。由于两侧时间常数相同，减弱侧与增强侧的绝对面积比为$\beta_E$。
- $\delta$：post相对pre的放电时间差。

**Case A：pre先、post后。** $\delta>0$，$F_E>0$，突触增强；相隔越久，增强越小。

**Case B：post先、pre后。** $\delta<0$，$F_E<0$，突触减弱；$\beta_E$越大，减弱一侧越强。

该规则用于**突触前神经元为E的连接，即E→E与E→I**。这里是成对时间差的STDP；没有引入Ruslim2025式triplet项。[^plasticity]

## 1.7 I-STDP：对称的抑制性学习窗

**原文式(13)，PDF第4页：**

$$
F_I(\delta)=
\frac{A_I}{1-\dfrac{\tau_{I1}}{\tau_{I2}}\beta_I}
\left[
\exp\!\left(-\frac{|\delta|}{\tau_{I1}}\right)
-\frac{\tau_{I1}}{\tau_{I2}}\beta_I
\exp\!\left(-\frac{|\delta|}{\tau_{I2}}\right)
\right].
\tag{13}
$$

**参数约束，原文式(14)：**

$$
\frac{\tau_{I1}}{\tau_{I2}}
<\min\!\left(\frac{1}{\beta_I},1\right).
\tag{14}
$$

其中：

- $F_I(\delta)$：一对pre／post spike对抑制性权重的改变。
- $A_I=0.02$：学习窗的强度参数；由式(13)可以直接得到$F_I(0)=A_I$。
- $\tau_{I1}=10\,\mathrm{ms}$：较快指数项的时间常数。
- $\tau_{I2}=20\,\mathrm{ms}$：较慢指数项的时间常数。
- $\beta_I$：调节增强与减弱平衡的参数。
- $|\delta|$：pre与post spike的时间距离，因此交换先后顺序不会改变学习窗的值。

**理解：** 第一项衰减较快，第二项衰减较慢。靠近$\delta=0$时，学习窗为正，抑制性连接增强；间隔较大时，较慢的负项可以占优势，连接减弱。式(14)保证$\tau_{I1}<\tau_{I2}$且分母为正。$\beta_I$改变两部分的相对贡献；由于它们共同构成双指数差，不能不加计算就将$\beta_I$直接当成实际正负面积之比。

该规则用于**突触前神经元为I的连接，即I→E与I→I**。两种STDP分别更新各自连接的$w$，共同影响网络最终处于亚临界、临界或超临界状态。[^plasticity]

# 2.测量网络状态与记忆所用指标

## 2.1 文中如何定义一次雪崩

**先把所有神经元的spike合成一条按时间排序的序列，再按相邻spike的间隔分组。**

原文在一段10 min自发活动中，取得合并后的spike时间序列：

$$
\{t_1,t_2,\ldots,t_M\},\qquad t_1\leq t_2\leq\cdots\leq t_M.
$$

以这条合并序列的平均spike间隔作为连接事件的时间尺度。把“平均间隔”写成公式就是（定义的显式改写）：

$$
\delta_{\mathrm{av}}
=\frac{1}{M-1}\sum_{j=2}^{M}(t_j-t_{j-1}).
$$

原文也用$\Delta t$表示这个间隔；这里改为$\delta_{\mathrm{av}}$，与$0.1\,\mathrm{ms}$数值模拟步长区分。

一串spike$\{t_i,\ldots,t_{i+n-1}\}$内部满足：

$$
t_j-t_{j-1}<\delta_{\mathrm{av}},
\qquad j=i+1,\ldots,i+n-1.
$$

也就是，相邻两个spike的间隔小于该平均间隔时，将它们连接为同一事件；较长的间隙将事件分开。原文用严格小于号描述事件内部条件，没有另列恰好等于阈值时的端点处理。

**雪崩大小：事件内spike总数。** 把原文的文字定义写成：

$$
S_k=|A_k|=n_k.
$$

其中$A_k$是第$k$个事件的spike集合，$n_k$是其spike数量。同一个神经元在一个事件内多次放电，贡献多个spike。原文没有在这一方法段给出雪崩持续时间$T$的独立计算公式，因此不自行补成固定bin数或某个加一端点公式。[^avalanche]

需要与两种算法区分：

- 本文直接比较相邻spike间隙，没有先建立固定时间网格再找连续非空bin。
- 本文没有采用[[Cortical Circuit Dynamics Are Homeostatically Tuned to Criticality In Vivo|Ma2019]]的群体活动第35百分位阈值分割。

## 2.2 Criticality index：$\Delta Cr$

作者对雪崩大小分布在log–log坐标中做线性回归，得到拟合幂律$p_{\mathrm{fit}}(s)$；再将其与经验分布$p_{\mathrm{emp}}(s)$比较。

线性回归与幂律之间的关系可以写成（说明用等价式）：

$$
\log p_{\mathrm{fit}}(s)=a_0+a_1\log s,
\qquad
p_{\mathrm{fit}}(s)=C s^{a_1}.
$$

其中$s$是雪崩大小，$a_0$是截距，$a_1$是斜率，$C$为相应比例常数；对数的底保持一致即可。

**原先的偏差指标，原文式(15)，PDF第4页：**

$$
\Delta p=
\sum_{s=s_{\min}}^{s_{\max}}
\left[p_{\mathrm{emp}}(s)-p_{\mathrm{fit}}(s)\right].
\tag{15}
$$

其中，$s_{\min}$、$s_{\max}$是用于比较的大小范围上下限。经验分布在拟合曲线上方时贡献正值，在下方时贡献负值，两侧可能相互抵消。因此，$\Delta p$接近0时，分布仍可能明显弯曲。

**为减少这种误判，原文式(16)、(17)，PDF第5页：**

$$
A_{\mathrm{upper}}=
\sum_{s=s_{\min}}^{s_{\max}}
\max\!\left[p_{\mathrm{emp}}(s)-p_{\mathrm{fit}}(s),0\right].
\tag{16}
$$

$$
A_{\mathrm{lower}}=
\sum_{s=s_{\min}}^{s_{\max}}
\min\!\left[p_{\mathrm{emp}}(s)-p_{\mathrm{fit}}(s),0\right].
\tag{17}
$$

其中，$A_{\mathrm{upper}}\geq0$累计上方偏差，$A_{\mathrm{lower}}\leq0$累计下方偏差。虽然拟合是在log–log坐标完成，这两个式子相减的是**概率值本身**，不应改写成log概率的残差。

**最终指标，原文式(18)：**

$$
\Delta Cr=
\begin{cases}
A_{\mathrm{upper}},
&|A_{\mathrm{upper}}|\geq|A_{\mathrm{lower}}|,\\[3pt]
A_{\mathrm{lower}},
&|A_{\mathrm{upper}}|<|A_{\mathrm{lower}}|.
\end{cases}
\tag{18}
$$

即**选择绝对值更大的一侧，并保留其正负号**；原文这个定义没有额外分母或归一化项。

作者按以下规则分类：

$$
\begin{aligned}
\Delta Cr<0&:\quad\text{subcritical},\\
\Delta Cr\approx0&:\quad\text{critical},\\
\Delta Cr>0&:\quad\text{supercritical}.
\end{aligned}
$$

数值模拟中$s_{\max}=100$，即神经元总数；培养实验方法按记录电极数设上限。$s_{\min}$通过最小化线性拟合的平方误差选择。**这个拟合上限不代表一次事件最多只能有100个spike。** 每个条件的$\Delta Cr$均值和标准差由30个初始种子得到。[^avalanche]

**$\Delta Cr$与Ma2019的DCC不是同一个指标。** 前者描述雪崩大小分布相对拟合幂律的偏离，后者比较大小、持续时间指数预测的关系与实测size–duration关系。本论文主要依赖$\Delta Cr$分类，不能把大小分布接近幂律单独提升为已经完整证明临界性。

## 2.3 EI balance与自发放电率

数值模拟中，作者记录每个神经元受到的E、I突触电流，取2 s自发活动计算两者的cross-correlation，并比较平均E输入、平均I输入和I/E幅度比。

**用于计算的两类电流就是式(1)中的突触项（原文第5页说明）：**

$$
I_E(t)=(E_{\mathrm{exc}}-v(t))g_{\mathrm{exc}}(t),
\qquad
I_I(t)=(E_{\mathrm{inh}}-v(t))g_{\mathrm{inh}}(t).
$$

其中$I_E$、$I_I$为此处说明用的电流记号。按膜电位方程的符号，正常电位范围内$I_I$通常为负；比较I/E强度时应区分抑制性电流的负号与输入幅度。原文在该方法段没有给出cross-correlation的完整归一化公式及实现细节，因此这里不补造一个指定版本。

**理解：** EI balance强调两类输入在时间上的耦合，不能仅理解为E和I神经元数量相同、输入逐点完全相等，或I/E必须为1。超临界网络也可以有很高的E/I相关性，所以EI balance不能单独代替临界性判断。

另需区分：$f_{\mathrm{rest}}$是式(4)中指定的**静息噪声发放参数**；连接形成后实际测到的**网络平均自发放电率**还受到循环输入和可塑性影响，两者不等同。[^ei]

## 2.4 Fading memory：刺激身份还能被读出多久

对四种刺激模式，观察每次测试刺激之后的网络活动。将活动按$B=20\,\mathrm{ms}$分箱，用所有100个神经元的放电率构成100维向量，再由sparse logistic regression（SLR，稀疏逻辑回归）预测刚才呈现的是哪一种刺激。

把原文的特征构造写成公式（说明用改写）：

$$
r_i(k)=\frac{N_i(k)}{B},
\qquad
\mathbf r(k)=[r_1(k),\ldots,r_{100}(k)]^{\mathsf T}.
$$

其中$N_i(k)$是第$i$个神经元在第$k$个bin中的spike数，$r_i(k)$是该bin的放电率；$B$以秒计时，放电率单位为Hz。培养实验用记录电极的特征向量，维数随所选电极数变化，不能直接视为100个已分离单细胞。

同样把分类准确率和机会水平写成：

$$
\mathrm{Accuracy}(k)
=\frac{\text{第}k\text{个bin分类正确的测试trial数}}
{\text{参与该bin测试的trial总数}},
\qquad
p_{\mathrm{chance}}=\frac14=0.25.
$$

SLR的具体损失函数和正则化超参数没有在这段主文中完整列出，笔记不为它补写一个未经核对的训练公式。

**Decay time：** 测试刺激后，解码准确率**首次回到机会水平的时间**。操作上逐bin用二项检验判断正确数是否显著高于机会水平（$p<0.05$）。它不等于所有显著bin的总时长；较晚再出现显著bin时，不能把分散的片段直接相加。

模型测试每种刺激有40个不同随机种子的trial，每次测试前先自由运行10 s。准确率使用10组分层shuffle交叉验证；判断机会水平的二项检验另外使用leave-one-out结果。前后decay time用Wilcoxon检验，逐bin前后准确率用Friedman检验。

**两个时间尺度需要分开：** 训练后等待12 h／24 h，是网络经历自发活动的时间；fading memory时长，是等待结束后重新给一个测试刺激，其身份在随后毫秒尺度响应中还能被读出多久。“24 h后测得记忆增强”不能写成一次刺激的活动或信息连续持续了24 h。[^memory]

# 3.通过网络模型进行的实验以及结论

## 实验1：仅靠噪声与突触可塑性，能否形成不同网络状态

**实验目的：** 从没有有效突触强度的初态出发，检查自发放电与STDP能否形成稳定网络，以及什么参数让网络接近临界。

**Step 1：建立初态。** 使用80E＋20I的全连接网络，将所有权重设为0，默认静息噪声放电率为0.4 Hz。初始放电只能先由escape noise产生。

**Step 2：让网络自由发展。** 保留STP和E／I-STDP，让网络自发运行72 h，使连接和活动充分稳定；每个参数条件使用30个不同初始随机种子。

**Step 3：改变学习窗的增强／减弱平衡。** 比较不同$\beta_E$和$\beta_I$下的连接、raster、雪崩大小分布及E／I输入。图中代表参数为：

- Crt：$\beta_E=1$，$\beta_I=1.15$。
- Sub：$\beta_E=1.2$，$\beta_I=1.2$。
- Sup：$\beta_E=1$，$\beta_I=1$。

**实验结果：**

1. 减弱相对增强更占优势时，形成的网络偏亚临界，大规模群体burst较少，抑制性输入很弱。
2. 增强相对占优势时，网络偏超临界，大规模burst较多，活动大小的多样性较低。
3. 合适的学习窗可形成接近幂律的雪崩大小分布，本文将其归为临界网络。
4. 临界和超临界网络均可具有较高的E／I输入相关性；亚临界网络的抑制通路较弱。因此，**有EI balance仍需继续判断活动分布与动力学状态**。

**结论：** 在这套模型中，噪声提供最初的随机spike，STDP塑造连接；噪声与学习窗共同决定网络形成的状态，不能概括为“只要有噪声，就必然自组织到临界”。

**对应图：Fig.1A–D，PDF第5页；代表参数与30次数值模拟见Fig.2图注，第7页；结果见第8页。**[^exp1]

## 实验2：重复刺激破坏临界状态后，自发活动能否让它恢复

**实验目的：** 检查临界性是否只存在于没有输入的初始条件，还是能在经历学习扰动后重新恢复。

**Step 1：选取已经稳定的网络。** 在实验1形成的Crt、Sub、Sup网络上记录刺激前的$\Delta Cr$作为基线。

**Step 2：创建四种空间刺激。** 每种刺激选择6个E神经元，同时使这些神经元的膜电位增加30 mV。四种模式对应不同的空间输入。

**Step 3：反复呈现。** 四种模式随机交替，总体以1 Hz呈现；**每种900次、总计3600次，持续1 h**。这里不是四种模式各自同时以1 Hz运行，否则总时长会被误算。

**Step 4：停止这轮重复输入。** 让网络继续带噪声自发运行12 h，并保持可塑性过程；每小时用10 min自发活动计算雪崩分布和$\Delta Cr$。

**Step 5：与刺激前比较。** 比较Crt的恢复过程，也比较Sub和Sup是否更接近0。Fig.2A另外展示了两轮1 h刺激扰动，用于观察重复适应；不能据此把其余所有分析都写成两轮训练。

**实验结果：**

- Crt的$\Delta Cr$在重复刺激期间暂时降低，刺激结束后约2 h回到初始水平。
- Sub受到扰动后，$\Delta Cr$先降低再升高，稳定后的值比原先更接近0。
- Sup也能在刺激及后续自发活动后向0靠近。
- 在测试参数范围中，12 h后的$|\Delta Cr|$总体小于刺激前，说明这些网络经过这一流程后，按本文指标更接近临界。

**结论：** 学习引起的扰动与后续自发重整可以改变网络的稳定状态。这里的“稳态”来自噪声、E-STDP、I-STDP及短时资源动力学的共同作用，论文没有像Ma2019那样另加入一个显式以目标放电率计算的synaptic scaling公式。

**对应图：Fig.2A–B，PDF第7页；刺激协议见第4页，结果解释见第8页。**[^exp2]

## 实验3：临界性变化时，基线放电和EI输入是否也恢复

**实验目的：** 在同一刺激—自发活动流程中，判断$\Delta Cr$的变化是否伴随活动基线和突触输入的重整。

**Step 1：采用实验2的四模式、1 h重复刺激。** 不另外增加一种刺激方法。

**Step 2：比较三个时点。** 对比刺激前pre、刺激刚结束0 h、以及12 h自发活动之后；基线放电还按小时追踪。

**Step 3：同时测量多项读出。** 计算网络自发放电率、E／I电流cross-correlation、平均E输入、平均I输入和I/E比，避免只凭平均放电率推断临界性。

**实验结果：**

1. 三类网络的自发放电基线都在刺激刚结束时下降。
2. 随后Crt的基线回到原水平；Sub的基线升到原水平以上，Sup的基线降到原水平以下。因此，Sub和Sup的调整涉及稳定水平的改变，不能统一写成全部变量都精确回到原值。
3. Crt的E／I输入相关性暂时下降，12 h后恢复；E和I平均输入也先下降再恢复，而I/E比相对稳定。
4. Sub和Sup的EI耦合会改变，但并非所有读出都回到刺激前的水平。

**结论：** 网络可以在经历扰动后重新稳定，同时改变活动基线。**本实验改变的是重复输入，随后观测网络平均自发放电率的变化；它没有把模型参数$f_{\mathrm{rest}}$逐小时人为设成这些测得的平均放电率。**

**对应图：Fig.2C–D，PDF第7页；结果见第8页。**[^exp3]

## 实验4：重复刺激与12 h自发活动，能否增强特定输入的fading memory

**实验目的：** 检查状态恢复后，网络是否仍然保留对曾反复出现的模式更强、更持久的响应，而不只是恢复一张与经验无关的连接图。

**Step 1：训练前测试。** 向Crt、Sub、Sup网络分别呈现四种测试模式，记录每次刺激之后的神经元活动。

**Step 2：执行训练和等待。** 按实验2，每种模式900次、总计3600次；之后让网络自发运行12 h。

**Step 3：用相同模式重新测试。** 每种模式使用40个随机trial，按20 ms统计各神经元放电率。

**Step 4：解码刺激身份。** 用SLR读出各bin中四分类的准确率，按第2.4节的首次回到机会水平判据确定fading-memory decay time，再比较pre与12 h。

**实验结果：**

- 训练前，Sub的fading memory可以比Crt更长。**临界网络并非在任何训练阶段、任何读出条件下都最好。**
- 训练加12 h自发活动后，三类网络的诱发响应和解码时长均有变化，其中Crt的增强最明显，Sup的变化很小。
- 训练后的Crt解码时长超过Sub，说明训练前的相对排序发生了改变。
- 图中测试响应在随后约0.2 s范围内逐渐回落；decay time主要在毫秒尺度，不是12 h。

**结论：** 在本文的四模式任务与参数条件下，临界附近的网络更有利于把重复经验转化为更持久的刺激特异性响应。不能把平均放电率衰减得慢、整体活动变强和刺激身份可解码得更久当成同一指标。

**对应图：Fig.3A–D，PDF第9页；解码方法见第5–6页，结果见第10页。**[^exp4]

## 实验5：记忆增强是刺激结束就完成，还是在自发活动期间继续发展

**实验目的：** 把“重复刺激当下的改变”和“其后数小时的变化”分开，观察网络是否在没有继续训练输入时发生重整。

**Step 1：固定重复刺激流程。** 重点使用Crt网络，完成3600次输入后停止这轮重复刺激。

**Step 2：逐小时重新测试。** 在0至12 h的不同等待时点，用四种模式测量诱发放电率、分类准确率和decay time。

**Step 3：比较时间曲线。** 不只比较pre和12 h两个端点，还看改善是逐渐出现还是出现明显跃迁；Sub和Sup的对应结果由正文援引补充图说明。

**实验结果：**

- Crt在最初约1 h内变化较缓，随后诱发响应和decay time出现较明显的阶跃式上升，并在后续等待期维持较高水平。
- Sub的变化更渐进，Sup总体变化较小。

**结论：** 网络中可测的刺激特异性响应在自发活动阶段仍会发展，训练当下的响应不能代表最终效果。不过，主文这一组结果主要是时间过程比较，不能单独证明“只有自发活动这一个因素”造成了变化；STDP和短时突触机制也在模型中共同运行。

**对应图：Fig.3E–F，PDF第9页；结果及补充图说明见第10页。**[^exp5]

## 实验6：重复次数越多，之后的刺激特异性响应是否越强

**实验目的：** 检查增强是否依赖输入经验的数量，避免把它解释为只要等待足够久就会无条件发生。

**Step 1：改变训练量。** 将总重复输入数从400增加到3600；每种刺激模式出现次数相等。Fig.3G–H还给出未重复训练的0次基线，图示比较点为0、400、1200、2000、2800、3600次。

**Step 2：在相应训练流程之后测量响应。** 比较相同四类测试刺激的分类准确率曲线与decay time。

**Step 3：以重复次数为横轴比较。** 观察分类与记忆增强是否随次数线性增加，还是存在明显转折区间。

**实验结果：** 随着重复次数增加，fading-memory decay time总体上升，呈类似sigmoid的变化；少量重复的效果有限，在一定范围内增强更明显，之后趋于平台。

**结论：** 后续增强与先前的重复经验有关。这里改变的是总刺激次数；由于总体刺激频率为1 Hz，训练时长也随次数改变，不能把它当成已经独立控制了“次数”和“训练总时长”的实验。

**对应图：Fig.3G–H，PDF第9页；重复量范围及结果见第6、10页。**[^exp6]

## 实验7：哪些突触变化保留较久，噪声水平会不会影响保留

**实验目的：** 寻找记忆增强对应的连接变化，并检查噪声是否只带来有益作用。

**Step 1：追踪训练前后的权重。** 比较重复刺激刚结束与12 h后各类突触的$w$，分清暂时增强和长期保留。

**Step 2：比较三种状态。** 看Crt、Sub、Sup中兴奋性和抑制性连接的变化是否相同。

**Step 3：比较噪声条件。** 正文进一步援引补充图S2，比较适当噪声与较高噪声下的记忆和权重保留；**16页主文没有列出这组比较的完整噪声档位及改参细节，不在笔记中补造数值。**

**实验结果：**

- Crt中，原文记号$w_{EE}$约6%的连接从接近0增强到接近1；这批增强连接中，约85%在12 h后仍不低于0.5。这里85%的分母是“被增强的连接”，不是全体连接。
- 兴奋性突触中保留较久的变化，与增强的刺激响应相伴；许多抑制性突触的增强更短暂。
- 较高噪声下，兴奋性权重更容易暂时增强，但较不容易长期保留；抑制性权重则更容易长期保留。正文同时指出，fading memory的延长出现在适当水平的自发活动条件下。

**结论：** 这套模型需要噪声驱动，但不能推出噪声越强越好。权重轨迹支持“部分兴奋性连接保留经验、抑制性连接参与调节”的解释；主文未逐一关闭这些连接组验证必要性，因此不能把观察到的权重变化直接写成已被独立干预证实的唯一记忆机制。

**对应结果：正文第10页，引用Supplementary Fig.S2、S4–S6；这些百分比来自主文对补图的文字报告。**[^exp7]

# 4.文中所做的培养神经元实验

## 4.1 使用了什么生物材料，怎样记录和刺激

使用 **8份培养28–37天的离散神经元培养物**，来源为**胚胎第18天（E18）的Wistar大鼠皮层**。这是离体培养的大鼠皮层网络，不能写成小鼠、成年动物或在体V1记录；8份培养物也不能直接改写成8只动物。[^culture]

**Step 1：制备培养物。** 取出E18大鼠皮层组织，用0.25% trypsin–EDTA在37 °C处理20 min后解离；将约38,000个细胞接种到高密度CMOS MEA的3.85×2.10 mm²记录区域，继续培养。胰酶用于细胞制备，**不是用来检验记忆或临界性机制的药理干预**。

**Step 2：选择记录电极。** MaxOne系统有26,400个电极，可同时读出其中1024个；作者先扫描spike信号，再按负峰幅度选择最多1024个记录位置。信号作300–3000 Hz带通滤波，以每个电极噪声标准差的−5倍检测事件。

**Step 3：选择刺激位置。** 根据spike波形负峰到正峰的间隔，将峰间时长不小于0.5 ms的电极视为位于putative excitatory neurons附近；这是波形分类，不能等同于分子标记确认的兴奋性细胞。

**Step 4：建立四种空间模式。** 选择24个刺激电极，使其覆盖测试区域、彼此至少间隔100 μm；每6个邻近位置组成一种模式，组内同时刺激。电刺激为正相在前的双相波，每相200 μs，幅度±200 mV。

**Step 5：保持培养环境。** 实验开始前半天换液，实验期间保留培养液；后续比较使用同一培养物刺激前后的记录。

**对应方法：PDF第6页，§2.2.1–2.2.2；电极与模式见Fig.4A，第11页。**[^culture]

## 实验1：重复电刺激后，四种模式是否变得更容易区分

**实验目的：** 在真实神经网络中检查，训练后经过一段自发活动，是否也能出现刺激特异性响应增强。

**Step 1：训练前测试。** 对四种电极模式分别给测试刺激，记录诱发的多电极时空活动作为pre。

**Step 2：反复训练。** 四种模式以随机顺序、总体1 s一个刺激的频率出现；每种900次，总计3600次，协议与模型对应。

**Step 3：让培养物经历自发活动间隔。** 在训练结束后0.5、3、24、48 h重新测试，不能把这些时点误写成同一次刺激的连续响应时长。

**Step 4：取得测试样本。** 每个测试时点，每种模式呈现20次，模式随机、相邻测试刺激间隔17–34 s；这样比较的是刺激后即时响应中保留的身份信息。

**Step 5：排除刺激伪迹并解码。** 去掉刺激后1 ms内的数据和被刺激饱和的电极数据，再按20 ms统计各记录电极放电率，使用SLR完成四分类。这里的特征来自电极记录，不能直接当作已分离出的单细胞放电率。

**实验结果：**

1. 24 h后，部分电极对特定模式的诱发响应增强，出现更多延迟的响应事件。
2. 解码准确率随训练后的等待过程改善，在24 h时最明显；四分类机会水平仍是0.25。
3. 24 h的decay time相对pre显著延长（Wilcoxon signed-rank，$p=0.038$）。
4. 到48 h，记忆时长回到接近原水平；这与模型在其测试期间维持增强有所不同。

**结论：** 真实培养网络能够在训练后一天的重新测试中表现出更持久的刺激特异性响应。这支持本文关于“经验之后的网络重整与记忆表达”的部分预测，但没有证明一段电活动持续保存刺激身份24 h。

本组主要是同一培养物的pre／post比较；主文没有报告以药物抑制自发活动、关闭STDP或阻断抑制性突触的对照，不能据此单独认定自发活动的必要性，或把模型里的每一条突触机制都视为已在培养物中验证。

**对应图：Fig.4B–D，PDF第11页；方法第6、8页，结果第10–12页。**[^culturememory]

## 实验2：记忆增强是否只是网络整体变得更兴奋

**实验目的：** 用同一批培养记录中的基线读出，检查训练后所有活动是否都普遍升高。

**Step 1：沿用实验1的训练与测试时点。** 仍比较pre、0.5、3、24、48 h。

**Step 2：测量自发活动基线。** 将没有测试刺激时的自发放电水平，与刺激后的诱发响应和分类结果分开。

**Step 3：比较两类变化。** 看decay time增强是否同时伴随显著的整体基线增高；正文还援引补充图S8，用放电率与刺激模式间的mutual information作辅助信息量分析。

**实验结果：** 基线自发放电率在这些测量时点没有显著变化，而24 h的刺激特异性解码改善。因此，诱发响应增强不能简单归结为培养网络的整体活动都显著抬高。

**结论：** 培养实验支持“特定输入的表达改善”，但它没有复现模型中的全部动力学过程。作者在讨论中明确承认：培养物没有观察到模型里“重复刺激暂时降低基线／偏离临界，再由自发活动恢复”的两阶段变化；Fig.4也没有给出对应的EI电流相关性或$\Delta Cr$恢复曲线。

**对应图：Fig.4E，PDF第11页；与模拟不一致的说明见第11–12页。**[^culturebaseline]

## 原文与定位

[论文页面](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2025.1585087/full) · [本地PDF](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf)。本文按2025年发表、2026-02-11更正的16页PDF核对；页码均为该文件的实际页序。

相关笔记：[[Synaptic Plasticity Enables Adaptive SelfTuning Critical Networks]]；[[Cortical Circuit Dynamics Are Homeostatically Tuned to Criticality In Vivo]]。

[^model]: PDF第2–4页，式(1)–(6)、Table 1及模型设置。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=2)。
[^plasticity]: PDF第3–4页，式(7)–(14)，资源恢复／消耗、事件更新、E／I-STDP。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=3)。
[^avalanche]: PDF第4–5页，雪崩分组、S、式(15)–(18)。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=4)。
[^ei]: PDF第5页，EI balance方法；第8、12页解释状态差异。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=5)。
[^memory]: PDF第5–6页，SLR、交叉验证、decay time与20 ms bin。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=6)。
[^exp1]: PDF第4、5、7、8页，72 h发展、三态与参数。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=8)。
[^exp2]: PDF第4、7、8页，刺激协议与临界性恢复，Fig.2A–B。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=7)。
[^exp3]: PDF第7–8页，基线和EI输入变化，Fig.2C–D。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=8)。
[^exp4]: PDF第9–10页，pre／12 h解码，Fig.3A–D。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=9)。
[^exp5]: PDF第9–10页，逐小时记忆变化，Fig.3E–F。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=10)。
[^exp6]: PDF第6、9、10页，重复次数与记忆，Fig.3G–H。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=9)。
[^exp7]: PDF第10页对补图S2、S4–S6的文字报告；未将未读补图当作直接核对来源。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=10)。
[^culture]: PDF第6页，培养物、MEA及电刺激方法；Fig.4A在第11页。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=6)。
[^culturememory]: PDF第6、8、10–12页，培养解码与decay time，Fig.4B–D。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=11)。
[^culturebaseline]: PDF第11–12页，基线、Fig.4E及培养实验未复现两阶段恢复的说明。[原文](file:///C:/Users/%E5%90%89%E6%AF%85%E8%AF%9A/Desktop/papers/2025_Frontiers%20in%20Neural%20Circuits_Emergent%20Functions%20of%20Noise-driven%20Spontaneous%20Activity%20-%20Homeostatic%20Maintenance%20of%20Criticality%20and%20Memory%20Consolidation.pdf#page=12)。
