利用朴素贝叶斯实现对RSS源上的文档进行分类，实在不忍吐槽RSS有效源地址真的好难找，即使找到了资源也不多，（可能是我不会使用吧。。。），全程很简单，对资源先做预处理，其中占用了代码的很多篇幅，实际分类代码很简单，
  
    gnb = NB.GaussianNB()  
    # bnb = NB.BernoulliNB()
    # mnb = NB.MultinomialNB()
    y_predict = gnb.fit(x_train, y_train).predict(x_test)
    return (y_predict != y_test).sum()
   便是全部核心分类代码。
