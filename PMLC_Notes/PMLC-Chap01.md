## Chapter 01 监督学习
>   1.1 简介    
    1.2 数据预处理  
    1.3 标记编码方法    
    1.4 线性回归器  
    1.5 回归的准确性    
    1.6 保存数据模型    
    1.7 岭回归器    
    1.8 多项式回归器    
    1.9 e.g.估算房屋价格    
    1.10 计算特征的重要度   
    1.11 e.g.共享单车的需求分布 
***
### 1.1 简介
需要的库：
    numpy; scipy; scikitlearn; matplot;
***
### 1.2 数据预处理
#### 1.2.1 准备工作
#### 1.2.2 多种处理方式
1. 均值移除 Mean Removal
    (z-score; zero-mean normalize)  
    使新的集合表现为：
    $$
        \mu = 0; \sigma = 1;
    $$
    算法：
    $$
        x^* = (x-\mu)/\sigma
    $$
    目的: 消除不同尺度特征间的bias  

2. 范围缩放 Scaling 
    (m-M normalization)
    使新的集合处于m到M的区间内
    算法（0-1 normalization）：
    $$
        x^* = 0 + (x-x_{min})/(X_{max}-X_{min})*(1-0)
    $$

3.  归一化 Normalization
    使特征向量统一到单位向量 （L1范数）     
    算法：
    $$
        x^* = x/(\Sigma X) \\
        s.t.\;|X| = 1
    $$

4. 二值化 Binarization
    ......
5. 独热编码 One-Hot Encoding    
    针对离散的非数字数据（如文本分类）
    使离散分类的特征“距离”变得更加合理

### 1.3 标记编码方法
* 编码顺序： 字母顺序
* 编号从0开始
* 创建编码对象[LE_OBJ]：preprocessing.LabelEncoder() 
* 编码对象成员方法[LE_OBJ.]:
  1. fit(INPUT_classes)  此处input_classes为待编码的列表，通常为文字标签列表
  2. classes_ 按字母顺序排列的存储的标签列表
  3. transform(input_label) 转成编码编号
  4. inverse_transform(index) 转回对应标签

### 1.4 线性回归器
* 普通最小二乘法(Ordinary Least Square - OLS)
* 数据格式：
    $$
        (\vec{x}_{i}, y_{i})
    $$
* 目标函数:
    $$
        \hat{f}(x)=\beta_0+\vec{\beta}\vec{x}\\
        first order:\; \hat{f}(x)=\beta_0+\beta_{1}x
    $$
* 标准损失函数：
    $$
        Q(\beta_0, \beta_1)=\sum^{n}_{i=1} [y_i-(\beta_0+\beta_1x_i)]^2
    $$
* 回归的度量：
    $$
        Sum of Square for Total - SST:\sum(y_i-\bar{y})^2\\
        Sum of Square for Regression - SSR:\sum(\hat{y}-\bar{y})^2\\
        Sum of Square for Error - SSE:\sum(y_i-\hat{y})^2
    $$
#### 1.4.2操作
1. 读取数据文件+新建存储对象    
   '''python    
        xt,yt=[flaote(i),  for i in line.split(',')]    
   '''
2. 拆分训练数据和测试数据   
    将x样本塑性为纵向向量   
    '''python   
        x_train/test = np.array(X[:traning_edge]).reshape(row#,col#)    
        y_train/test = np.array(X[:traning_edge])   
    ''' 
3. 训练与测试   
   "from sklearn importlinear_model"    
   新建线性模型对象:"obj_name=linear_model.LinearRegression()"  
   "obj_name.fit(x_data,y_data)"
   "y_predict = obj_name.predict(x_input)    
4. 作图 
   
### 1.5 回归的准确性
以下也可以是损失函数
1.  平均绝对误差  Mean Absolute Error (MAE)
    $$
        MAE = 1/n*(\sum^{n}_{i=1}|y_i-\hat{y}_i|)
    $$ 
2. Mean Square Error (MSE)
    $$
        MSE = 1/n*\sum^{n}_{i=1}(y_i-\hat{y}_i)^2
    $$
3. Median Absolute Error (MedAE)
    $$
        MedAE=|med(\hat{Y})-med(Y)|
    $$
4. Explained Variance Score
    $$
        EVS = 1 - \frac{Var(Y_i-\hat{Y}_i)}{Var(Y)}
    $$
    the closer to 1 the better
5. R-square score
    $$
        R^2 = 1-\frac{\sum^{n}_{i=1}(y_i-\hat{y}_i)^2}{\sum^{n}_{i=1}(y_i-\bar{y})^2}
    $$
    the closer to 1 the better  

### 1.6 保存数据模型
    需要的库:
        python2: cPickle
        python3: pickle
    open(arg1, arg2):
        arg1: somefile
        arg2: use wb/rb instead of w/r to write/read file
    
### 1.7 岭回归器
>消除少量异常值对整体线性回归的影响. 

标准损失函数：
$$
    \sum^{n}_{i=1}(y_i-\beta_i-\beta_1x_i)^2+\lambda\sum^{p}_{j=1}\beta_j^2\\
    \lambda\; should\;be\;manually\;chosen\; ti\; fit\; the\; sensitivity
$$
新建对象：
"ridge_regressor=linear_model.Ridge(alpha=?.??,fit_intercept=T/F, max_iter=???)"
"ridge_regressor.fit(x,y)"
"ridge_regressor.predict(x)"

### 1.8 多项式回归器
### 1.9 e.g.估算房屋价格
### 1.10 计算特征的重要度
### 1.11 e.g.共享单车的需求分布
