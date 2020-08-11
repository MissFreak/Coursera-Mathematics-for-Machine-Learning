主成分分析（PCA）是统计学和数据科学中一项重要的技术，本篇文章将从“What”  “How” “Why”等角度，介绍PCA方法的基本原理、优缺点、以及数学原理。 

# 背景知识 

读者最好熟悉以下内容： 


* 矩阵运算/线性代数：矩阵乘法、矩阵换位、矩阵求逆、矩阵分解、特征向量/特征值； 

* 统计/机器学习：标准化、方差、协方差、独立性、线性回归、特征选择。 
# 一、什么是PCA？ 

当我们的数据集中包含大量错综复杂的变量时，你可能会想：如何处理变量之间的关系？模型会过拟合吗？如何减少特征空间的维度，只集中在其中几个变量上面？ 

这时我们需要降维！ 

## 降维：特征消除与特征提取 

降维的方法有很多种，大致分为两类： 


* **特征消除** ：去掉一些变量，只保留相对重要的变量。这样简单易行，且保持变量的可解释性，但损失了去除的变量对模型的贡献。 

* **特征提取** ：将若干旧变量以特定方式组合，建立新变量，并根据它们对因变量的预测程度排序。 
## PCA的优缺点 

作为 一种特征提取技术 ，PCA的优点： 


* 实现降噪和去冗余，保留所有变量中最有价值的部分！ 

* 每个“新”变量彼此独立，满足线性回归模型的假设。 

缺点： 


* 可解释性较差，因为主成分是无实际意义的初始变量的线性组合。 
# 二、PCA的基本原理 

PCA的核心在于，计算一组特征向量和特征值，来总结变量之间的关系。其中，特征向量代表若干线性无关（正交）的方向，特征值代表每个方向的信息量（即方差），标志着每个方向的重要程度。 

如图，红色和绿色的方向代表了两个垂直的特征向量。显然，红色的方向比绿色的方向更重要，涵盖了更多信息。 

![图片](https://uploader.shimo.im/f/8SumQaoCxm19RbfY.png!thumbnail)

这些由初始变量线性组合而成的特征向量，就是数据的主成分（PC）。 

因此，主成分分析是一种信息的组织方式：把初始变量中大部分信息压缩到第一个主成分，剩余的最大信息放入第二个成分，以此类推。 

几何意义上，我们可以把原始数据集理解成m维空间中的n个点，这些点具有某种分布趋势，即在某个方向上分散开。主成分分析就是旋转坐标系到合适的位置，使得这个方向成为新的坐标轴。在主成分分析中，第一条主成分解释了最大的方差量，这意味着数据点沿该轴线的分布最离散，涵盖数据最大的信息。它提供了查看和评估数据的最佳角度，让人更为直观地看到数据中的差异。 

![图片](https://uploader.shimo.im/f/eq1FkwOBNPMOkMRj.png!thumbnail)

*Percentage of Variance (Information) for each by PC* 

根据信息量将主成分排序后，我们可以删除“最不重要”的主成分，将数据压缩或投影到更小的空间中。这样既降低了特征空间的维度，又在模型中保留了所有原始变量！ 

# 三、算法及操作步骤 

操作之前，要明确我们的数据呈现在n行、m+1列的表格中，分别代表n个sample、m个自变量（特征）的矩阵X，加上最后一列的因变量（Y）。 

![图片](https://uploader.shimo.im/f/khknl5psYgjGqzbv.png!thumbnail)

## 第1步：标准化得到新矩阵 X 

PCA之前，一个重要的步骤是执行标准化。因为PCA对初始变量的方差非常敏感。如果初始变量的范围（scale）比例悬殊，会影响结果（例如，范围在0到100的变量会主导范围介于0到1的变量）。因此，标准化能够保证每个变量的贡献相同。 

![图片](https://uploader.shimo.im/f/NpPc112OmNCtzftA.png!thumbnail)

注：standardization又称feature scaling 

方法：将原矩阵的每个元素，减去每个维度（列）上的平均值并除以标准差，使得每列均值为0、标准差为1，得到新矩阵X。 

![图片](https://uploader.shimo.im/f/Cz75tkOAXQqwaWRS.png!thumbnail)

## 第2步：计算协方差矩阵 XᵀX 

这一步的目的是了解变量之间的关系，因为高度相关的变量会包含冗余信息。为了识别这些相关性，需要计算协方差矩阵 (covariance matrix)。 

### 协方差 

![图片](https://uploader.shimo.im/f/CzrEIARdDLHdawaj.png!thumbnail)

### 协方差矩阵 

协方差矩阵是一个m×m对称矩阵（其中m是维数），其元素表示所有变量对儿的协方差，代表变量之间的关联性。协方差符号为正，则变量正相关；符号为负，变量负相关。 

![图片](https://uploader.shimo.im/f/MsmRy2AqiIY0eiE0.png!thumbnail)

注：对角线上其实表示方差 (variance) 

### 求解协方差矩阵 

由于第一步中，我们已经减去了平均值 *μ* ，因此直接计算 cov = XᵀX 即可，得到对角线对称的 半正定矩阵。 

## 第3步：计算 XᵀX 的特征向量矩阵P 

基本原理中介绍了， **主成分就是协方差矩阵的特征向量！而每个主成分的方差量，就是附加在特征向量上的系数，即特征值！** 

### 特征向量和特征值 

![图片](https://uploader.shimo.im/f/iid2dIWACVl0Hfll.png!thumbnail)

上述等式 意味着， 对于特征向量v ， 矩阵 A 的 应用 只会拉伸或缩放其长度 λ 倍 ， 而不改变其方向。 

### **求解方法** 

变换等式，计算行列式，求出一组 λ ，继而求出对应的v。 

![图片](https://uploader.shimo.im/f/UhFFvjIoDcNt6XEo.png!thumbnail)

### 得到 投影矩阵 P 

前面我们计算得到m个m维特征向量和特征值。按特征值的顺序，降序排列特征向量，就可以得到按重要性排序的主成分。 

![图片](https://uploader.shimo.im/f/mtnAX8FStZZPUFIU.png!thumbnail)

为了计算每个成分所占的方差（信息）百分比，可以将每个成分的特征值除以特征值之和。 

我们可以选择保留或舍弃特征值较低的分量，把剩下的特征向量组成一个特征向量矩阵，又称转换矩阵或者投影矩阵（m，p），其中p代表保留的主成分的数量。 

![图片](https://uploader.shimo.im/f/QehsN949lLl3XJCZ.png!thumbnail)

## 第4步：重塑数据XP 

最后，将数据从原始轴重新定向到由主成分表示的轴，转换到新的坐标空间。 

实现方法：数据集X（n，m）点乘投影矩阵（m，p）。 

最终，我们将原始的m维数据集变为p维矩阵（n，p）。 

![图片](https://uploader.shimo.im/f/2GFYMY8OWwM0DTNd.png!thumbnail)

# 四、深入思考一些问题 

## 为什么PCA用到协方差矩阵 

我们知道，协方差矩阵代表了变量之间的相关性。协方差矩阵主对角线上的元素是各个维度上的方差，其他元素是两两维度间的协方差。 

那么，要让新维度之间相关性减到最低，就需要协方差矩阵中非对角线元素都为零（矩阵对角化）。 

我们知道，矩阵A可以对角化的话，可以通过相似矩阵进行下面的特征值分解： 

![图片](https://uploader.shimo.im/f/WvM8fXPSgOuiW0q0.png!thumbnail)

![图片](https://uploader.shimo.im/f/qFOWtotOrB5Sbulk.png!thumbnail)

图：来自马同学 

因此，需要找到一个转换矩阵P（特征向量矩阵），将原数据集 X 投射到 X '： X ' = XP，并且满足新的协方差矩阵 X ' X ' ᵀ为对角矩阵， 对角线上是特征值，其他元素为零。 

分解一下： X ' X ' ᵀ = (P X ) (P X ) ᵀ = (PX)( Xᵀ Pᵀ ) = P( X Xᵀ ) Pᵀ 

这个式子揭示了原协方差矩阵和现对角矩阵的关系。经过P变换后的现对角矩阵，只剩下每个新维度上的方差，而所有其他协方差皆为零。 

这不正是PCA要达到的目的吗？找出若干线性无关的主成分！ 



**参考文献：** 

[https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)

[https://builtin.com/data-science/step-step-explanation-principal-component-analysis](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

[https://setosa.io/ev/principal-component-analysis/](https://setosa.io/ev/principal-component-analysis/)

