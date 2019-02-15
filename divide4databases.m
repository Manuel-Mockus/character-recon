load('./mnist-original.mat')

data1 = uint8.empty(784,0);
label1 = double.empty;

data2 = uint8.empty(784,0);
label2 = double.empty;

data3 = uint8.empty(784,0);
label3 = double.empty;

data4 = uint8.empty(784,0);
label4 = double.empty;

k = 1;
while k <= 70000
    label1 = [label1,label(k)];
    data1 = [data1,data(:,k)];

    label2 = [label2,label(k+1)];
    data2 = [data2,data(:,k+1)];

    label3 = [label3,label(k+2)];
    data3 = [data3,data(:,k+2)];

    label4 = [label4,label(k+3)];
    data4 = [data4,data(:,k+3)];
    
    k = k+4;
end

%disp(size(data1));
%disp(size(data2));
disp(size(data3));
disp(size(data4));

disp(size(label1));
disp(size(label2));
disp(size(label3));
disp(size(label4));

n = 3;

I = find(label1==n);

mydigit = reshape(data1(:,I(5) ),[28,28])';

mydigit = 255-mydigit;

imshow(mydigit)

data = 0;
data = data1;
label = 0;
label = label1;
save('dataset_Jonathan.mat','data','label');
data = 0;
data = data2;
label = 0;
label = label2;
save('dataset_Manuel.mat','data','label');
data = 0;
data = data3;
label = 0;
label = label3;
save('dataset_Marin.mat','data','label');
data = 0;
data = data4;
label = 0;
label = label4;
save('dataset_Octave.mat','data','label');


