# C++ new delete malloc free 区别 #
+ new、delete是c++中的操作符，malloc、free是C中的一个函数，它们都可用于申请动态内存和释放内存。  
+ new 不止是分配内存，而且会调用类的构造函数，同理delete会调用类的析构函数，而malloc则只分配内存，不会进行初始化类成员的工作，同样free也不会调用析构函数。由于malloc/free是库函数而不是运算符，不在编译器控制权限之内，不能够把执行构造函数和析构函数的任务强加于malloc/free。  
+ 内存泄漏对于malloc或者new都可以检查出来的，区别在于new可以指明是哪个文件的哪一行，而malloc没有这些信息。   
+ new的效率malloc稍微低一些，new可以认为是malloc加构造函数的执行。new出来的指针是直接带类型信息的。 而malloc返回的都是void指针。  
+ malloc不会抛异常，而new会；无法重定义malloc失败时的默认行为（返回NULL）,但是我们可以重定义new失败时默认行为，比如不让其抛出异常。  


## malloc()函数 ##
**1.1 malloc的全称是memory allocation，中文叫动态内存分配。**
  
    原型：extern void *malloc(unsigned int num_bytes)
   
说明：分配长度为num_bytes字节的内存块。如果分配成功则返回指向被分配内存的指针，分配失败返回空指针NULL。当内存不再使用时，应使用free()函数将内存块释放。
  
**1.2 void *malloc(int size)**   
说明：malloc 向系统申请分配指定size个字节的内存空间，返回类型是 void* 类型。void* 表示未确定类型的指针。C,C++规定，void* 类型可以强制转换为任何其它类型的指针。  
备注：void* 表示未确定类型的指针，更明确的说是指申请内存空间时还不知道用户是用这段空间来存储什么类型的数据（比如是char还是int或者...）  

**1.3 free**
  
    void free(void *FirstByte);  
 该函数是将之前用malloc分配的空间还给程序或者是操作系统，也就是释放了这块内存，让它重新得到自由。
  
**1.4 注意事项**
  
1）申请了内存空间后，必须检查是否分配成功。  
2）当不需要再使用申请的内存时，记得释放；释放后应该把指向这块内存的指针指向NULL，防止程序后面不小心使用了它。  
3）这两个函数应该是配对。如果申请后不释放就是内存泄露；如果无故释放那就是什么也没有做。释放只能一次，如果释放两次及两次以上会出现错误（释放空指针例外，释放空指针其实也等于啥也没做，所以释放空指针释放多少次都没有问题）。  
4）虽然malloc()函数的类型是(void *),任何类型的指针都可以转换成(void *),但是最好还是在前面进行强制类型转换，因为这样可以躲过一些编译器的检查。
  
**1.5 malloc()到底从哪里得到了内存空间？**  

答案是从堆里面获得空间。也就是说函数返回的指针是指向堆里面的一块内存。操作系统中有一个记录空闲内存地址的链表。当操作系统收到程序的申请时，就会遍历该链表，然后就寻找第一个空间大于所申请空间的堆结点，然后就将该结点从空闲结点链表中删除，并将该结点的空间分配给程序。
  
## new运算符 ##
**2.1 C++中，用new和delete动态创建和释放数组或单个对象。**  

动态创建对象时，只需指定其数据类型，而不必为该对象命名，new表达式返回指向该新创建对象的指针，我们可以通过指针来访问此对象。
  
    int *pi=new int;
  
这个new表达式在堆区中分配创建了一个整型对象，并返回此对象的地址，并用该地址初始化指针pi 。
  
**2.2 动态创建对象的初始化**  
动态创建的对象可以用初始化变量的方式初始化。  

    int *pi=new int(100);   //指针pi所指向的对象初始化为100
    string *ps=new string(10,'9');  //*ps 为“9999999999”

如果不提供显示初始化，对于类类型，用该类的默认构造函数初始化；而内置类型的对象则无初始化。  
也可以对动态创建的对象做值初始化：  

	int *pi=new int( );//初始化为0  
	int *pi=new int;//pi 指向一个没有初始化的int  
	string *ps=new string( );//初始化为空字符串 （对于提供了默认构造函数的类类型，没有必要对其对象进行值初始化）
  
**2.3 撤销动态创建的对象**
  
    delete表达式释放指针指向的地址空间。  
    delete pi;   // 释放单个对象  
    delete [ ]pi;//释放数组
  
如果指针指向的不是new分配的内存地址，则使用delete是不合法的。
  
**2.4 在delete之后，重设指针的值**  

delete p; //执行完该语句后，p变成了不确定的指针，在很多机器上，尽管p值没有明确定义，但仍然存放了它之前所指对象的地址，然后p所指向的内存已经被释放了，所以p不再有效。此时，该指针变成了悬垂指针（悬垂指针指向曾经存放对象的内存，但该对象已经不存在了）。悬垂指针往往导致程序错误，而且很难检测出来。  
一旦删除了指针所指的对象，立即将指针置为0，这样就非常清楚的指明指针不再指向任何对象。（零值指针：int *ip=0;） 
 
**2.5 区分零值指针和NULL指针**  

零值指针，是值是0的指针，可以是任何一种指针类型，可以是通用变体类型void*也可以是char*，int*等等。  
空指针，其实空指针只是一种编程概念，就如一个容器可能有空和非空两种基本状态，而在非空时可能里面存储了一个数值是0，因此空指针是人为认为的指针不提供任何地址讯息。  

**2.6 new分配失败时，返回什么？**  
1993年前，c++一直要求在内存分配失败时operator   new要返回0，现在则是要求operator   new抛出std::bad_alloc异常。很多c++程序是在编译器开始支持新规范前写的。c++标准委员会不想放弃那些已有的遵循返回0规范的代码，所以他们提供了另外形式的operator   new(以及operator   new[])以继续提供返回0功能。这些形式被称为“无抛出”，因为他们没用过一个throw，而是在使用new的入口点采用了nothrow对象:   

    class   widget   {   ...   };  
    widget   *pw1   =   new   widget;//   分配失败抛出std::bad_alloc  
    if   (pw1   ==   0)   ... //   这个检查一定失败  
    widget   *pw2   =   new   (nothrow)   widget;   //   若分配失败返回0  
    if   (pw2   ==   0)   ... //   这个检查可能会成功  

## malloc和new的区别 ##
**3.1 new 返回指定类型的指针，并且可以自动计算所需要大小。**   

    int *p; 　　
    p = new int; //返回类型为int* 类型(整数型指针)，分配大小为 sizeof(int); 　　
    或： 　　
    int* parr; 　　
    parr = new int [100]; //返回类型为 int* 类型(整数型指针)，分配大小为 sizeof(int) * 100; 　

而 malloc 则必须要由我们计算字节数，并且在返回后强行转换为实际类型的指针。

    int* p; 　　
    p = (int *) malloc (sizeof(int)*128);//分配128个（可根据实际需要替换该数值）整型存储单元，并将这128个连续的整型存储单元的首地址存储到指针变量p中  
    double *pd=(double *) malloc (sizeof(double)*12);//分配12个double型存储单元，并将首地址存储到指针变量pd中

**3.2 malloc 只管分配内存，并不能对所得的内存进行初始化，所以得到的一片新内存中，其值将是随机的。**  
除了分配及最后释放的方法不一样以外，通过malloc或new得到指针，在其它操作上保持一致。

## 有了malloc/free为什么还要new/delete？ ##

+ malloc与free是C++/C语言的标准库函数，new/delete是C++的运算符。它们都可用于申请动态内存和释放内存。  
+ 对于非内部数据类型的对象而言，光用maloc/free无法满足动态对象的要求。对象在创建的同时要自动执行构造函数，对象在消亡之前要自动执行析构函数。由于malloc/free是库函数而不是运算符，不在编译器控制权限之内，不能够把执行构造函数和析构函数的任务强加于malloc/free。

因此C++语言需要一个能完成动态内存分配和初始化工作的运算符new，以及一个能完成清理与释放内存工作的运算符delete。注意new/delete不是库函数。

我们不要企图用malloc/free来完成动态对象的内存管理，应该用new/delete。由于内部数据类型的“对象”没有构造与析构的过程，对它们而言malloc/free和new/delete是等价的。
3) 既然new/delete的功能完全覆盖了malloc/free，为什么C++不把malloc/free淘汰出局呢？这是因为C++程序经常要调用C函数，而C程序只能用malloc/free管理动态内存。

如果用free释放“new创建的动态对象”，那么该对象因无法执行析构函数而可能导致程序出错。如果用delete释放“malloc申请的动态内存”，结果也会导致程序出错，但是该程序的可读性很差。所以new/delete必须配对使用，malloc/free也一样。



### 参考资料 ###
[http://lib.csdn.net/article/cplusplus/23837](http://lib.csdn.net/article/cplusplus/23837)
