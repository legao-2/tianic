clc
clear
close all
%Load the Titanic data and show it
Train = readtable('titanic\train.csv');
Test = readtable('titanic\test.csv');
%��Train.Sex���ַ�����Ϊ��ֵ
femaleIndex = strcmp(Train.Sex,'female');
Temp = zeros(length(femaleIndex),1);%������Train.Sexͬά����
Temp(femaleIndex) = 1;
Train.Sex = num2cell(Temp);
%��Test.Sex���ַ�����Ϊ��ֵ
femaleIndexTe = strcmp(Test.Sex,'female');
Temp = zeros(length(femaleIndexTe),1);%������Test.Sexͬά����
Temp(femaleIndexTe) = 1;
Test.Sex = num2cell(Temp);
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);
%���հ������滻Ϊƽ��ֵ
avgAge = nanmean(Train.Age); 
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 
%�����仮������
Train.Age=round(Train.Age/10);
Test.Age=round(Test.Age/10);


%��ȡ����
trainData=Train{:,[3 5 6]};
testData=Test{:,[2 4 5]};
%��ȡ���ݱ�ǩ
trainLabel=Train{:,2};
trainSampleNumber=size(trainLabel,1);
% testLabel=Test{:,2};
attributeNumber=size(trainData,2);
attributeValueNumber=length(unique(Train.Age));
%����ÿ������������ĸ���
labelProbability=tabulate(trainLabel);
%P_yi,����P(yi)
P_y1=labelProbability(1,3)/100;
P_y2=labelProbability(2,3)/100;
%%
%
count_1=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=0����£���i������ȡjֵ������ͳ��
count_2=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=1����£���i������ȡjֵ������ͳ��
%ͳ��ÿһ��������ÿ��ȡֵ������
for jj=1:2
    for j=1:trainSampleNumber
        for ii=1:attributeNumber
            for k=0:attributeValueNumber
                if jj==1
                    if trainLabel(j,1)==0&&trainData(j,ii)==k
                        count_1(ii,k+1)=count_1(ii,k+1)+1;%������������������Ϊ�㣬��������ͳһ��һ
                    end
                else
                    if trainLabel(j,1)==1&&trainData(j,ii)==k
                        count_2(ii,k+1)=count_2(ii,k+1)+1;
                    end
                end
            end
        end
    end
end
%�����i������ȡjֵ�ĸ��ʣ�P_a_y1�Ƿ���Ϊy=1ǰ����ȡֵ�������������ơ�
%ȡ������˹ƽ��
P_a_y1=(count_1+1)./(labelProbability(1,2)+attributeValueNumber);
P_a_y2=(count_2+1)./(labelProbability(2,2)+attributeValueNumber);

%%
%ʹ�ò��Լ��������ݲ���
labelPredictNumber=zeros(2,1);
predictLabel=zeros(size(testData,1),1);
for kk=1:size(testData,1)
    testDataTemp=testData(kk,:)+1;%���ϱ���������λ����һ��
    Pxy1=1;
    Pxy2=1;   
    %����P��x|yi��
    for iii=1:attributeNumber
        Pxy1=Pxy1*P_a_y1(iii,testDataTemp(iii));
        Pxy2=Pxy2*P_a_y2(iii,testDataTemp(iii));       
    end
    %����P(x|yi)*P(yi)
    PxyPy1=P_y1*Pxy1;
    PxyPy2=P_y2*Pxy2;
    if PxyPy1>PxyPy2
        predictLabel(kk,1)=0;
        %disp(['this item belongs to No.',num2str(1),' label or the victim label'])
        labelPredictNumber(1,1)=labelPredictNumber(1,1)+1;
    else
        predictLabel(kk,1)=1;
        labelPredictNumber(2,1)=labelPredictNumber(2,1)+1;
        %disp(['this item belongs to No.',num2str(2),' label or the survivor label'])
    end
end
% testLabelCount=tabulate(testLabel);
% �����������
% disp('the confusion matrix is : ')
% C_Bayes=confusionmat(testLabel,predictLabel);
PassengerId=[1:length(predictLabel)]';
colName={'PassengerId','Survived'};
resultTable=table(PassengerId,predictLabel,'VariableNames',colName);
writetable(resultTable,'submission.csv');
%%
%ʹ���Դ���Ҷ˹�㷨���ֲ��Լ�
Mdl=fitcnb(trainData,trainLabel,...
    'ClassNames',{'0','1'});
Plabel=predict(Mdl,testData);
writetable(table(PassengerId,Plabel,'VariableNames',colName),'cmpsub.csv');
Cmpar=str2num(cell2mat(Plabel))==predictLabel;

