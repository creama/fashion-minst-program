# ����
ʹ�������򵥵��㷨��FashionMNIST����ѵ���Ͳ��ԡ���һ���㷨��һ���Ĳ��ȫ�������磬�ڶ����㷨����LeNet-5�Ļ�������������������㣬
�ҳ�ΪLeNet-7��ȫ��������Ĵ�����model_bp�ļ��У�LeNet-7������model_cnn�С�
# FashionMNIST���ݼ�
FashionMNIST���ݼ��ĵ�ַ��https://github.com/zalandoresearch/fashion-mnist��
Fashion MNIST���ݼ��ǵ¹��о�����Zalando Research��2017��8�·ݣ���Github���Ƴ���һ���������ݼ���
����ѵ��������60000�����������Լ�����10000����������Ϊ10�࣬ÿһ�������ѵ�����������Ͳ�������������ͬ��ÿ����������28��28�ĻҶ�ͼ��
����10���ǩ������ÿ���������и���Ψһ�ı�ǩ��

# ѵ��
ѵ�����̵�epoch��Ϊ30��batch_size��Ϊ100��ѵ��������Ὣ���Ե�top1׼ȷ�ʡ�top2׼ȷ�ʺ�loss���߻�������
## ѵ��ȫ��������
��model_bpĿ¼��
```python
python train_bp.py
```
## ѵ��LeNet-7
��model_cnnĿ¼��
```python
python train_cnn.py
```

# ����
���Թ��̻�Ӳ������������ȡ��batch_size���������в��ԣ�����Ԥ���label��ӡ������
## ����ȫ��������
��model_bpĿ¼��
```python
python test_bp.py
```
## ����LeNet-7
��model_cnnĿ¼��
```python
python test_cnn.py
```