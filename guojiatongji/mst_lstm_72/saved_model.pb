??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	MLCConv2D

input"T
filter"T

unique_key"T*num_args
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)
"
	transposebool( "
num_argsint(
?
MLCLSTM
x"T
hidden_size
output_size
kernel"T
recurrent_kernel"T	
bias"T
output"T"
Ttype:
2"
return_sequencesbool( "
dropoutfloat"
bidirectionalbool( "

activationstringtanh
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02&tf_macos-v0.1-alpha2-AS-67-gf3595294ab8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:`0*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:0*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:<H*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:H*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
lstm_2/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?**
shared_namelstm_2/lstm_cell_4/kernel
?
-lstm_2/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_4/kernel*
_output_shapes
:	0?*
dtype0
?
#lstm_2/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*4
shared_name%#lstm_2/lstm_cell_4/recurrent_kernel
?
7lstm_2/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_4/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_2/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_2/lstm_cell_4/bias
?
+lstm_2/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_4/bias*
_output_shapes	
:?*
dtype0
?
lstm_3/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_namelstm_3/lstm_cell_5/kernel
?
-lstm_3/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_5/kernel*
_output_shapes
:	?*
dtype0
?
#lstm_3/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*4
shared_name%#lstm_3/lstm_cell_5/recurrent_kernel
?
7lstm_3/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_5/recurrent_kernel*
_output_shapes
:	<?*
dtype0
?
lstm_3/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_namelstm_3/lstm_cell_5/bias
?
+lstm_3/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_5/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:`0*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/m
?
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:<H*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:H*
dtype0
?
 Adam/lstm_2/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*1
shared_name" Adam/lstm_2/lstm_cell_4/kernel/m
?
4Adam/lstm_2/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_4/kernel/m*
_output_shapes
:	0?*
dtype0
?
*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m
?
>Adam/lstm_2/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_4/bias/m
?
2Adam/lstm_2/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_4/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_3/lstm_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_3/lstm_cell_5/kernel/m
?
4Adam/lstm_3/lstm_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_5/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*;
shared_name,*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m
?
>Adam/lstm_3/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m*
_output_shapes
:	<?*
dtype0
?
Adam/lstm_3/lstm_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_5/bias/m
?
2Adam/lstm_3/lstm_cell_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_5/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:`0*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/v
?
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:<H*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:H*
dtype0
?
 Adam/lstm_2/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*1
shared_name" Adam/lstm_2/lstm_cell_4/kernel/v
?
4Adam/lstm_2/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_4/kernel/v*
_output_shapes
:	0?*
dtype0
?
*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*;
shared_name,*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v
?
>Adam/lstm_2/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_2/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_2/lstm_cell_4/bias/v
?
2Adam/lstm_2/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_4/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/lstm_3/lstm_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" Adam/lstm_3/lstm_cell_5/kernel/v
?
4Adam/lstm_3/lstm_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_5/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*;
shared_name,*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v
?
>Adam/lstm_3/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v*
_output_shapes
:	<?*
dtype0
?
Adam/lstm_3/lstm_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/lstm_3/lstm_cell_5/bias/v
?
2Adam/lstm_3/lstm_cell_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_5/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?t
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?s
value?sB?s B?s
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
layer_with_weights-4
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
 
R
)trainable_variables
*	variables
+regularization_losses
,	keras_api
R
-trainable_variables
.	variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
R
7trainable_variables
8	variables
9regularization_losses
:	keras_api
R
;trainable_variables
<	variables
=regularization_losses
>	keras_api
R
?trainable_variables
@	variables
Aregularization_losses
B	keras_api

C	keras_api

D	keras_api

E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
R
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api

T	keras_api
l
Ucell
V
state_spec
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
h

[kernel
\bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
l
acell
b
state_spec
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
h

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?1m?2m?Jm?Km?[m?\m?gm?hm?rm?sm?tm?um?vm?wm?v?v?#v?$v?1v?2v?Jv?Kv?[v?\v?gv?hv?rv?sv?tv?uv?vv?wv?
?
0
1
#2
$3
14
25
J6
K7
r8
s9
t10
[11
\12
u13
v14
w15
g16
h17
?
0
1
#2
$3
14
25
J6
K7
r8
s9
t10
[11
\12
u13
v14
w15
g16
h17
 
?
xnon_trainable_variables
ymetrics
zlayer_regularization_losses
{layer_metrics
trainable_variables

|layers
	variables
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
}non_trainable_variables
~metrics
layer_regularization_losses
?layer_metrics
trainable_variables
?layers
 	variables
!regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
%trainable_variables
?layers
&	variables
'regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
)trainable_variables
?layers
*	variables
+regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
-trainable_variables
?layers
.	variables
/regularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
3trainable_variables
?layers
4	variables
5regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
7trainable_variables
?layers
8	variables
9regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
?layers
<	variables
=regularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
@	variables
Aregularization_losses
 
 
 
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ftrainable_variables
?layers
G	variables
Hregularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ltrainable_variables
?layers
M	variables
Nregularization_losses
 
 
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ptrainable_variables
?layers
Q	variables
Rregularization_losses
 
?

rkernel
srecurrent_kernel
tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 

r0
s1
t2

r0
s1
t2
 
?
?non_trainable_variables
?states
?metrics
 ?layer_regularization_losses
?layer_metrics
Wtrainable_variables
?layers
X	variables
Yregularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
]trainable_variables
?layers
^	variables
_regularization_losses
?

ukernel
vrecurrent_kernel
wbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 

u0
v1
w2

u0
v1
w2
 
?
?non_trainable_variables
?states
?metrics
 ?layer_regularization_losses
?layer_metrics
ctrainable_variables
?layers
d	variables
eregularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
itrainable_variables
?layers
j	variables
kregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_2/lstm_cell_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_2/lstm_cell_4/recurrent_kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElstm_2/lstm_cell_4/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElstm_3/lstm_cell_5/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#lstm_3/lstm_cell_5/recurrent_kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUElstm_3/lstm_cell_5/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
?2
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
t2

r0
s1
t2
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
?	variables
?regularization_losses
 
 
 
 
 

U0
 
 
 
 
 

u0
v1
w2

u0
v1
w2
 
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
?	variables
?regularization_losses
 
 
 
 
 

a0
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_1_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_2_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_1_inputserving_default_conv2d_2_inputserving_default_conv2d_inputconv2d_1/kernelconv2d_1/biasconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasdense_4/kerneldense_4/biaslstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biasdense_5/kerneldense_5/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biasdense_6/kerneldense_6/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_47764
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_2/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_4/bias/Read/ReadVariableOp-lstm_3/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/v/Read/ReadVariableOpConst*N
TinG
E2C	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_48844
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biastotalcounttotal_1count_1total_2count_2Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/m Adam/lstm_2/lstm_cell_4/kernel/m*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mAdam/lstm_2/lstm_cell_4/bias/m Adam/lstm_3/lstm_cell_5/kernel/m*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mAdam/lstm_3/lstm_cell_5/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/v Adam/lstm_2/lstm_cell_4/kernel/v*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vAdam/lstm_2/lstm_cell_4/bias/v Adam/lstm_3/lstm_cell_5/kernel/v*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vAdam/lstm_3/lstm_cell_5/bias/v*M
TinF
D2B*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_49049Ҵ
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_47017

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48284

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_48008
inputs_0
inputs_1
inputs_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)dense_4_mlcmatmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&lstm_2_mlclstm_readvariableop_resource,
(lstm_2_mlclstm_readvariableop_1_resource,
(lstm_2_mlclstm_readvariableop_2_resource-
)dense_5_mlcmatmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&lstm_3_mlclstm_readvariableop_resource,
(lstm_3_mlclstm_readvariableop_1_resource,
(lstm_3_mlclstm_readvariableop_2_resource-
)dense_6_mlcmatmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/MLCMatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp? dense_5/MLCMatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/MLCMatMul/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp_1?lstm_2/MLCLSTM/ReadVariableOp_2?lstm_3/MLCLSTM/ReadVariableOp?lstm_3/MLCLSTM/ReadVariableOp_1?lstm_3/MLCLSTM/ReadVariableOp_2?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D	MLCConv2Dinputs_1&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d/BiasAdd?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D	MLCConv2Dinputs_2&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_2/BiasAdd?
activation_3/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_3/Relu?
activation_2/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_2/Relu?
activation_4/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_4/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMulmax_pooling2d/MaxPool:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul max_pooling2d_1/MaxPool:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMulactivation_4/Relu:activations:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mult
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate/concat?
 dense_4/MLCMatMul/ReadVariableOpReadVariableOp)dense_4_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_4/MLCMatMul/ReadVariableOp?
dense_4/MLCMatMul	MLCMatMulconcatenate/concat:output:0(dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_4/MLCMatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MLCMatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_4/BiasAddy
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_4/Tanh?
max_pooling2d_2/MaxPoolMaxPooldense_4/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
tf.compat.v1.squeeze/SqueezeSqueeze max_pooling2d_2/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeezeq
lstm_2/ShapeShape%tf.compat.v1.squeeze/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_2/Shape?
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack?
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1?
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/mul/y?
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/Less/y?
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/packed/1?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/mul/y?
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/Less/y?
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/packed/1?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeros_1{
lstm_2/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/MLCLSTM/hidden_size{
lstm_2/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/MLCLSTM/output_size?
lstm_2/MLCLSTM/ReadVariableOpReadVariableOp&lstm_2_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
lstm_2/MLCLSTM/ReadVariableOp?
lstm_2/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_2_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_2/MLCLSTM/ReadVariableOp_1?
lstm_2/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_2_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_2/MLCLSTM/ReadVariableOp_2?
lstm_2/MLCLSTMMLCLSTM%tf.compat.v1.squeeze/Squeeze:output:0#lstm_2/MLCLSTM/hidden_size:output:0#lstm_2/MLCLSTM/output_size:output:0%lstm_2/MLCLSTM/ReadVariableOp:value:0'lstm_2/MLCLSTM/ReadVariableOp_1:value:0'lstm_2/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_2/MLCLSTM?
 dense_5/MLCMatMul/ReadVariableOpReadVariableOp)dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_5/MLCMatMul/ReadVariableOp?
dense_5/MLCMatMul	MLCMatMullstm_2/MLCLSTM:output:0(dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_5/MLCMatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MLCMatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_5/BiasAddu
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_5/Tanh\
lstm_3/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/zeros_1z
lstm_3/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/MLCLSTM/hidden_sizez
lstm_3/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/MLCLSTM/output_size?
lstm_3/MLCLSTM/ReadVariableOpReadVariableOp&lstm_3_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_3/MLCLSTM/ReadVariableOp?
lstm_3/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_3_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02!
lstm_3/MLCLSTM/ReadVariableOp_1?
lstm_3/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_3_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_3/MLCLSTM/ReadVariableOp_2?
lstm_3/MLCLSTMMLCLSTMdense_5/Tanh:y:0#lstm_3/MLCLSTM/hidden_size:output:0#lstm_3/MLCLSTM/output_size:output:0%lstm_3/MLCLSTM/ReadVariableOp:value:0'lstm_3/MLCLSTM/ReadVariableOp_1:value:0'lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_3/MLCLSTM}
lstm_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_3/Reshape/shape?
lstm_3/ReshapeReshapelstm_3/MLCLSTM:output:0lstm_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/Reshape?
 dense_6/MLCMatMul/ReadVariableOpReadVariableOp)dense_6_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02"
 dense_6/MLCMatMul/ReadVariableOp?
dense_6/MLCMatMul	MLCMatMullstm_3/Reshape:output:0(dense_6/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_6/MLCMatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MLCMatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_6/BiasAddp
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_6/Tanh?
IdentityIdentitydense_6/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/MLCMatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/MLCMatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/MLCMatMul/ReadVariableOp^lstm_2/MLCLSTM/ReadVariableOp ^lstm_2/MLCLSTM/ReadVariableOp_1 ^lstm_2/MLCLSTM/ReadVariableOp_2^lstm_3/MLCLSTM/ReadVariableOp ^lstm_3/MLCLSTM/ReadVariableOp_1 ^lstm_3/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/MLCMatMul/ReadVariableOp dense_4/MLCMatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/MLCMatMul/ReadVariableOp dense_5/MLCMatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/MLCMatMul/ReadVariableOp dense_6/MLCMatMul/ReadVariableOp2>
lstm_2/MLCLSTM/ReadVariableOplstm_2/MLCLSTM/ReadVariableOp2B
lstm_2/MLCLSTM/ReadVariableOp_1lstm_2/MLCLSTM/ReadVariableOp_12B
lstm_2/MLCLSTM/ReadVariableOp_2lstm_2/MLCLSTM/ReadVariableOp_22>
lstm_3/MLCLSTM/ReadVariableOplstm_3/MLCLSTM/ReadVariableOp2B
lstm_3/MLCLSTM/ReadVariableOp_1lstm_3/MLCLSTM/ReadVariableOp_12B
lstm_3/MLCLSTM/ReadVariableOp_2lstm_3/MLCLSTM/ReadVariableOp_2:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?

?
B__inference_dense_6_layer_call_and_return_conditional_losses_47415

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_47338

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_46637
conv2d_input
conv2d_1_input
conv2d_2_input1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource3
/model_dense_4_mlcmatmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource0
,model_lstm_2_mlclstm_readvariableop_resource2
.model_lstm_2_mlclstm_readvariableop_1_resource2
.model_lstm_2_mlclstm_readvariableop_2_resource3
/model_dense_5_mlcmatmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource0
,model_lstm_3_mlclstm_readvariableop_resource2
.model_lstm_3_mlclstm_readvariableop_1_resource2
.model_lstm_3_mlclstm_readvariableop_2_resource3
/model_dense_6_mlcmatmul_readvariableop_resource1
-model_dense_6_biasadd_readvariableop_resource
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?$model/dense_4/BiasAdd/ReadVariableOp?&model/dense_4/MLCMatMul/ReadVariableOp?$model/dense_5/BiasAdd/ReadVariableOp?&model/dense_5/MLCMatMul/ReadVariableOp?$model/dense_6/BiasAdd/ReadVariableOp?&model/dense_6/MLCMatMul/ReadVariableOp?#model/lstm_2/MLCLSTM/ReadVariableOp?%model/lstm_2/MLCLSTM/ReadVariableOp_1?%model/lstm_2/MLCLSTM/ReadVariableOp_2?#model/lstm_3/MLCLSTM/ReadVariableOp?%model/lstm_3/MLCLSTM/ReadVariableOp_1?%model/lstm_3/MLCLSTM/ReadVariableOp_2?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2D	MLCConv2Dconv2d_1_input,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model/conv2d_1/BiasAdd?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2D	MLCConv2Dconv2d_input*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model/conv2d/BiasAdd?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2D	MLCConv2Dconv2d_2_input,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model/conv2d_2/BiasAdd?
model/activation_3/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation_3/Relu?
model/activation_2/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation_2/Relu?
model/activation_4/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation_4/Relu?
model/max_pooling2d_1/MaxPoolMaxPool%model/activation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/max_pooling2d/MaxPoolMaxPool%model/activation_2/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool?
model/tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/tf.math.multiply/Mul/y?
model/tf.math.multiply/MulMul$model/max_pooling2d/MaxPool:output:0%model/tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
model/tf.math.multiply/Mul?
model/tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
model/tf.math.multiply_1/Mul/y?
model/tf.math.multiply_1/MulMul&model/max_pooling2d_1/MaxPool:output:0'model/tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
model/tf.math.multiply_1/Mul?
model/tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2 
model/tf.math.multiply_2/Mul/y?
model/tf.math.multiply_2/MulMul%model/activation_4/Relu:activations:0'model/tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
model/tf.math.multiply_2/Mul?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2model/tf.math.multiply/Mul:z:0 model/tf.math.multiply_1/Mul:z:0 model/tf.math.multiply_2/Mul:z:0&model/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
model/concatenate/concat?
&model/dense_4/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_4_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02(
&model/dense_4/MLCMatMul/ReadVariableOp?
model/dense_4/MLCMatMul	MLCMatMul!model/concatenate/concat:output:0.model/dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
model/dense_4/MLCMatMul?
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp?
model/dense_4/BiasAddBiasAdd!model/dense_4/MLCMatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
model/dense_4/BiasAdd?
model/dense_4/TanhTanhmodel/dense_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
model/dense_4/Tanh?
model/max_pooling2d_2/MaxPoolMaxPoolmodel/dense_4/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_2/MaxPool?
"model/tf.compat.v1.squeeze/SqueezeSqueeze&model/max_pooling2d_2/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2$
"model/tf.compat.v1.squeeze/Squeeze?
model/lstm_2/ShapeShape+model/tf.compat.v1.squeeze/Squeeze:output:0*
T0*
_output_shapes
:2
model/lstm_2/Shape?
 model/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm_2/strided_slice/stack?
"model/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm_2/strided_slice/stack_1?
"model/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm_2/strided_slice/stack_2?
model/lstm_2/strided_sliceStridedSlicemodel/lstm_2/Shape:output:0)model/lstm_2/strided_slice/stack:output:0+model/lstm_2/strided_slice/stack_1:output:0+model/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm_2/strided_slicew
model/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros/mul/y?
model/lstm_2/zeros/mulMul#model/lstm_2/strided_slice:output:0!model/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_2/zeros/muly
model/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros/Less/y?
model/lstm_2/zeros/LessLessmodel/lstm_2/zeros/mul:z:0"model/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_2/zeros/Less}
model/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros/packed/1?
model/lstm_2/zeros/packedPack#model/lstm_2/strided_slice:output:0$model/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm_2/zeros/packedy
model/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_2/zeros/Const?
model/lstm_2/zerosFill"model/lstm_2/zeros/packed:output:0!model/lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model/lstm_2/zeros{
model/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros_1/mul/y?
model/lstm_2/zeros_1/mulMul#model/lstm_2/strided_slice:output:0#model/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_2/zeros_1/mul}
model/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros_1/Less/y?
model/lstm_2/zeros_1/LessLessmodel/lstm_2/zeros_1/mul:z:0$model/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_2/zeros_1/Less?
model/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_2/zeros_1/packed/1?
model/lstm_2/zeros_1/packedPack#model/lstm_2/strided_slice:output:0&model/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm_2/zeros_1/packed}
model/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_2/zeros_1/Const?
model/lstm_2/zeros_1Fill$model/lstm_2/zeros_1/packed:output:0#model/lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model/lstm_2/zeros_1?
 model/lstm_2/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/lstm_2/MLCLSTM/hidden_size?
 model/lstm_2/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2"
 model/lstm_2/MLCLSTM/output_size?
#model/lstm_2/MLCLSTM/ReadVariableOpReadVariableOp,model_lstm_2_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02%
#model/lstm_2/MLCLSTM/ReadVariableOp?
%model/lstm_2/MLCLSTM/ReadVariableOp_1ReadVariableOp.model_lstm_2_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02'
%model/lstm_2/MLCLSTM/ReadVariableOp_1?
%model/lstm_2/MLCLSTM/ReadVariableOp_2ReadVariableOp.model_lstm_2_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02'
%model/lstm_2/MLCLSTM/ReadVariableOp_2?
model/lstm_2/MLCLSTMMLCLSTM+model/tf.compat.v1.squeeze/Squeeze:output:0)model/lstm_2/MLCLSTM/hidden_size:output:0)model/lstm_2/MLCLSTM/output_size:output:0+model/lstm_2/MLCLSTM/ReadVariableOp:value:0-model/lstm_2/MLCLSTM/ReadVariableOp_1:value:0-model/lstm_2/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
model/lstm_2/MLCLSTM?
&model/dense_5/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model/dense_5/MLCMatMul/ReadVariableOp?
model/dense_5/MLCMatMul	MLCMatMulmodel/lstm_2/MLCLSTM:output:0.model/dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
model/dense_5/MLCMatMul?
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp?
model/dense_5/BiasAddBiasAdd!model/dense_5/MLCMatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model/dense_5/BiasAdd?
model/dense_5/TanhTanhmodel/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model/dense_5/Tanhn
model/lstm_3/ShapeShapemodel/dense_5/Tanh:y:0*
T0*
_output_shapes
:2
model/lstm_3/Shape?
 model/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 model/lstm_3/strided_slice/stack?
"model/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm_3/strided_slice/stack_1?
"model/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/lstm_3/strided_slice/stack_2?
model/lstm_3/strided_sliceStridedSlicemodel/lstm_3/Shape:output:0)model/lstm_3/strided_slice/stack:output:0+model/lstm_3/strided_slice/stack_1:output:0+model/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/lstm_3/strided_slicev
model/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model/lstm_3/zeros/mul/y?
model/lstm_3/zeros/mulMul#model/lstm_3/strided_slice:output:0!model/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_3/zeros/muly
model/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_3/zeros/Less/y?
model/lstm_3/zeros/LessLessmodel/lstm_3/zeros/mul:z:0"model/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_3/zeros/Less|
model/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
model/lstm_3/zeros/packed/1?
model/lstm_3/zeros/packedPack#model/lstm_3/strided_slice:output:0$model/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm_3/zeros/packedy
model/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_3/zeros/Const?
model/lstm_3/zerosFill"model/lstm_3/zeros/packed:output:0!model/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
model/lstm_3/zerosz
model/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model/lstm_3/zeros_1/mul/y?
model/lstm_3/zeros_1/mulMul#model/lstm_3/strided_slice:output:0#model/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/lstm_3/zeros_1/mul}
model/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/lstm_3/zeros_1/Less/y?
model/lstm_3/zeros_1/LessLessmodel/lstm_3/zeros_1/mul:z:0$model/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model/lstm_3/zeros_1/Less?
model/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
model/lstm_3/zeros_1/packed/1?
model/lstm_3/zeros_1/packedPack#model/lstm_3/strided_slice:output:0&model/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/lstm_3/zeros_1/packed}
model/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lstm_3/zeros_1/Const?
model/lstm_3/zeros_1Fill$model/lstm_3/zeros_1/packed:output:0#model/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
model/lstm_3/zeros_1?
 model/lstm_3/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2"
 model/lstm_3/MLCLSTM/hidden_size?
 model/lstm_3/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2"
 model/lstm_3/MLCLSTM/output_size?
#model/lstm_3/MLCLSTM/ReadVariableOpReadVariableOp,model_lstm_3_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#model/lstm_3/MLCLSTM/ReadVariableOp?
%model/lstm_3/MLCLSTM/ReadVariableOp_1ReadVariableOp.model_lstm_3_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02'
%model/lstm_3/MLCLSTM/ReadVariableOp_1?
%model/lstm_3/MLCLSTM/ReadVariableOp_2ReadVariableOp.model_lstm_3_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02'
%model/lstm_3/MLCLSTM/ReadVariableOp_2?
model/lstm_3/MLCLSTMMLCLSTMmodel/dense_5/Tanh:y:0)model/lstm_3/MLCLSTM/hidden_size:output:0)model/lstm_3/MLCLSTM/output_size:output:0+model/lstm_3/MLCLSTM/ReadVariableOp:value:0-model/lstm_3/MLCLSTM/ReadVariableOp_1:value:0-model/lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
model/lstm_3/MLCLSTM?
model/lstm_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
model/lstm_3/Reshape/shape?
model/lstm_3/ReshapeReshapemodel/lstm_3/MLCLSTM:output:0#model/lstm_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
model/lstm_3/Reshape?
&model/dense_6/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_6_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02(
&model/dense_6/MLCMatMul/ReadVariableOp?
model/dense_6/MLCMatMul	MLCMatMulmodel/lstm_3/Reshape:output:0.model/dense_6/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/dense_6/MLCMatMul?
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02&
$model/dense_6/BiasAdd/ReadVariableOp?
model/dense_6/BiasAddBiasAdd!model/dense_6/MLCMatMul:product:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model/dense_6/BiasAdd?
model/dense_6/TanhTanhmodel/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
model/dense_6/Tanh?
IdentityIdentitymodel/dense_6/Tanh:y:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/MLCMatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp'^model/dense_5/MLCMatMul/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp'^model/dense_6/MLCMatMul/ReadVariableOp$^model/lstm_2/MLCLSTM/ReadVariableOp&^model/lstm_2/MLCLSTM/ReadVariableOp_1&^model/lstm_2/MLCLSTM/ReadVariableOp_2$^model/lstm_3/MLCLSTM/ReadVariableOp&^model/lstm_3/MLCLSTM/ReadVariableOp_1&^model/lstm_3/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/MLCMatMul/ReadVariableOp&model/dense_4/MLCMatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2P
&model/dense_5/MLCMatMul/ReadVariableOp&model/dense_5/MLCMatMul/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2P
&model/dense_6/MLCMatMul/ReadVariableOp&model/dense_6/MLCMatMul/ReadVariableOp2J
#model/lstm_2/MLCLSTM/ReadVariableOp#model/lstm_2/MLCLSTM/ReadVariableOp2N
%model/lstm_2/MLCLSTM/ReadVariableOp_1%model/lstm_2/MLCLSTM/ReadVariableOp_12N
%model/lstm_2/MLCLSTM/ReadVariableOp_2%model/lstm_2/MLCLSTM/ReadVariableOp_22J
#model/lstm_3/MLCLSTM/ReadVariableOp#model/lstm_3/MLCLSTM/ReadVariableOp2N
%model/lstm_3/MLCLSTM/ReadVariableOp_1%model/lstm_3/MLCLSTM/ReadVariableOp_12N
%model/lstm_3/MLCLSTM/ReadVariableOp_2%model/lstm_3/MLCLSTM/ReadVariableOp_2:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48374
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48546
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
&__inference_lstm_2_layer_call_fn_48306

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
|
'__inference_dense_4_layer_call_fn_48216

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_471612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_47374

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46655

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_1_layer_call_fn_48132

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_470172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_47116

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_2_layer_call_fn_46673

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_466672
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_2_layer_call_fn_48171

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_470692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_46824

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48250

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_48147

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
~
F__inference_concatenate_layer_call_and_return_conditional_losses_47140

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_48593
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_469452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
#__inference_signature_wrapper_47764
conv2d_1_input
conv2d_2_input
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_1_inputconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_466372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input:^Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_46945

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_48152

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_470902
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46643

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48488

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_6_layer_call_and_return_conditional_losses_48615

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_48510

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_48189
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/2
?K
?
@__inference_model_layer_call_and_return_conditional_losses_47496
conv2d_input
conv2d_1_input
conv2d_2_input
conv2d_1_47437
conv2d_1_47439
conv2d_47442
conv2d_47444
conv2d_2_47447
conv2d_2_47449
dense_4_47464
dense_4_47466
lstm_2_47471
lstm_2_47473
lstm_2_47475
dense_5_47478
dense_5_47480
lstm_3_47483
lstm_3_47485
lstm_3_47487
dense_6_47490
dense_6_47492
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_47437conv2d_1_47439*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_470172"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_47442conv2d_47444*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_470432 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_47447conv2d_2_47449*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_470692"
 conv2d_2/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_470902
activation_3/PartitionedCall?
activation_2/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_471032
activation_2/PartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_471162
activation_4/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_466552!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_466432
max_pooling2d/PartitionedCallu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMul&max_pooling2d/PartitionedCall:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul(max_pooling2d_1/PartitionedCall:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul%activation_4/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mul?
concatenate/PartitionedCallPartitionedCalltf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_471402
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_47464dense_4_47466*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_471612!
dense_4/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_466672!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_47471lstm_2_47473lstm_2_47475*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472462 
lstm_2/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_5_47478dense_5_47480*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_472872!
dense_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0lstm_3_47483lstm_3_47485lstm_3_47487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473742 
lstm_3/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_6_47490dense_6_47492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_474152!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input
?
?
%__inference_model_layer_call_fn_48094
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_476722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_47103

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_47246

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
e
+__inference_concatenate_layer_call_fn_48196
inputs_0
inputs_1
inputs_2
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_471402
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/2
?
?
&__inference_lstm_3_layer_call_fn_48604
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_469922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_48207

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02	
BiasAdda
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_47604
conv2d_input
conv2d_1_input
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_1_inputconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_475652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_46992

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_2_layer_call_fn_48396
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_468242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_48137

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_47886
inputs_0
inputs_1
inputs_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)dense_4_mlcmatmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&lstm_2_mlclstm_readvariableop_resource,
(lstm_2_mlclstm_readvariableop_1_resource,
(lstm_2_mlclstm_readvariableop_2_resource-
)dense_5_mlcmatmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&lstm_3_mlclstm_readvariableop_resource,
(lstm_3_mlclstm_readvariableop_1_resource,
(lstm_3_mlclstm_readvariableop_2_resource-
)dense_6_mlcmatmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/MLCMatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp? dense_5/MLCMatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp? dense_6/MLCMatMul/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp_1?lstm_2/MLCLSTM/ReadVariableOp_2?lstm_3/MLCLSTM/ReadVariableOp?lstm_3/MLCLSTM/ReadVariableOp_1?lstm_3/MLCLSTM/ReadVariableOp_2?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2D	MLCConv2Dinputs_1&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_1/BiasAdd?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d/BiasAdd?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2D	MLCConv2Dinputs_2&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_2/BiasAdd?
activation_3/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_3/Relu?
activation_2/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_2/Relu?
activation_4/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_4/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMulmax_pooling2d/MaxPool:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul max_pooling2d_1/MaxPool:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMulactivation_4/Relu:activations:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mult
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2tf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate/concat?
 dense_4/MLCMatMul/ReadVariableOpReadVariableOp)dense_4_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_4/MLCMatMul/ReadVariableOp?
dense_4/MLCMatMul	MLCMatMulconcatenate/concat:output:0(dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_4/MLCMatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MLCMatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_4/BiasAddy
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_4/Tanh?
max_pooling2d_2/MaxPoolMaxPooldense_4/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool?
tf.compat.v1.squeeze/SqueezeSqueeze max_pooling2d_2/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeezeq
lstm_2/ShapeShape%tf.compat.v1.squeeze/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_2/Shape?
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stack?
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1?
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2?
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicek
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/mul/y?
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/Less/y?
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessq
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros/packed/1?
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/Const?
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeroso
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/mul/y?
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/Less/y?
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lessu
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/zeros_1/packed/1?
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/Const?
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_2/zeros_1{
lstm_2/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/MLCLSTM/hidden_size{
lstm_2/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_2/MLCLSTM/output_size?
lstm_2/MLCLSTM/ReadVariableOpReadVariableOp&lstm_2_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
lstm_2/MLCLSTM/ReadVariableOp?
lstm_2/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_2_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_2/MLCLSTM/ReadVariableOp_1?
lstm_2/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_2_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_2/MLCLSTM/ReadVariableOp_2?
lstm_2/MLCLSTMMLCLSTM%tf.compat.v1.squeeze/Squeeze:output:0#lstm_2/MLCLSTM/hidden_size:output:0#lstm_2/MLCLSTM/output_size:output:0%lstm_2/MLCLSTM/ReadVariableOp:value:0'lstm_2/MLCLSTM/ReadVariableOp_1:value:0'lstm_2/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_2/MLCLSTM?
 dense_5/MLCMatMul/ReadVariableOpReadVariableOp)dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_5/MLCMatMul/ReadVariableOp?
dense_5/MLCMatMul	MLCMatMullstm_2/MLCLSTM:output:0(dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_5/MLCMatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MLCMatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_5/BiasAddu
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_5/Tanh\
lstm_3/ShapeShapedense_5/Tanh:y:0*
T0*
_output_shapes
:2
lstm_3/Shape?
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stack?
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1?
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2?
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros/mul/y?
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros/Less/y?
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros/packed/1?
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/Const?
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros_1/mul/y?
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_3/zeros_1/Less/y?
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/zeros_1/packed/1?
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/Const?
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/zeros_1z
lstm_3/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/MLCLSTM/hidden_sizez
lstm_3/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_3/MLCLSTM/output_size?
lstm_3/MLCLSTM/ReadVariableOpReadVariableOp&lstm_3_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_3/MLCLSTM/ReadVariableOp?
lstm_3/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_3_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02!
lstm_3/MLCLSTM/ReadVariableOp_1?
lstm_3/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_3_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_3/MLCLSTM/ReadVariableOp_2?
lstm_3/MLCLSTMMLCLSTMdense_5/Tanh:y:0#lstm_3/MLCLSTM/hidden_size:output:0#lstm_3/MLCLSTM/output_size:output:0%lstm_3/MLCLSTM/ReadVariableOp:value:0'lstm_3/MLCLSTM/ReadVariableOp_1:value:0'lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_3/MLCLSTM}
lstm_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_3/Reshape/shape?
lstm_3/ReshapeReshapelstm_3/MLCLSTM:output:0lstm_3/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_3/Reshape?
 dense_6/MLCMatMul/ReadVariableOpReadVariableOp)dense_6_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02"
 dense_6/MLCMatMul/ReadVariableOp?
dense_6/MLCMatMul	MLCMatMullstm_3/Reshape:output:0(dense_6/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_6/MLCMatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MLCMatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_6/BiasAddp
dense_6/TanhTanhdense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_6/Tanh?
IdentityIdentitydense_6/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/MLCMatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/MLCMatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/MLCMatMul/ReadVariableOp^lstm_2/MLCLSTM/ReadVariableOp ^lstm_2/MLCLSTM/ReadVariableOp_1 ^lstm_2/MLCLSTM/ReadVariableOp_2^lstm_3/MLCLSTM/ReadVariableOp ^lstm_3/MLCLSTM/ReadVariableOp_1 ^lstm_3/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/MLCMatMul/ReadVariableOp dense_4/MLCMatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/MLCMatMul/ReadVariableOp dense_5/MLCMatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/MLCMatMul/ReadVariableOp dense_6/MLCMatMul/ReadVariableOp2>
lstm_2/MLCLSTM/ReadVariableOplstm_2/MLCLSTM/ReadVariableOp2B
lstm_2/MLCLSTM/ReadVariableOp_1lstm_2/MLCLSTM/ReadVariableOp_12B
lstm_2/MLCLSTM/ReadVariableOp_2lstm_2/MLCLSTM/ReadVariableOp_22>
lstm_3/MLCLSTM/ReadVariableOplstm_3/MLCLSTM/ReadVariableOp2B
lstm_3/MLCLSTM/ReadVariableOp_1lstm_3/MLCLSTM/ReadVariableOp_12B
lstm_3/MLCLSTM/ReadVariableOp_2lstm_3/MLCLSTM/ReadVariableOp_2:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_47161

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02	
BiasAdda
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
?
{
&__inference_conv2d_layer_call_fn_48113

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_470432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_48407

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
A__inference_conv2d_layer_call_and_return_conditional_losses_48104

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_47069

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_48176

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_46667

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_48051
inputs_0
inputs_1
inputs_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_475652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_46779

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:\ X
4
_output_shapes"
 :??????????????????0
 
_user_specified_nameinputs
?J
?
@__inference_model_layer_call_and_return_conditional_losses_47672

inputs
inputs_1
inputs_2
conv2d_1_47613
conv2d_1_47615
conv2d_47618
conv2d_47620
conv2d_2_47623
conv2d_2_47625
dense_4_47640
dense_4_47642
lstm_2_47647
lstm_2_47649
lstm_2_47651
dense_5_47654
dense_5_47656
lstm_3_47659
lstm_3_47661
lstm_3_47663
dense_6_47666
dense_6_47668
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_1_47613conv2d_1_47615*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_470172"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_47618conv2d_47620*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_470432 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_47623conv2d_2_47625*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_470692"
 conv2d_2/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_470902
activation_3/PartitionedCall?
activation_2/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_471032
activation_2/PartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_471162
activation_4/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_466552!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_466432
max_pooling2d/PartitionedCallu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMul&max_pooling2d/PartitionedCall:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul(max_pooling2d_1/PartitionedCall:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul%activation_4/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mul?
concatenate/PartitionedCallPartitionedCalltf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_471402
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_47640dense_4_47642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_471612!
dense_4/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_466672!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_47647lstm_2_47649lstm_2_47651*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472462 
lstm_2/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_5_47654dense_5_47656*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_472872!
dense_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0lstm_3_47659lstm_3_47661lstm_3_47663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473742 
lstm_3/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_6_47666dense_6_47668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_474152!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_47212

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_46661

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_466552
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48452

inputs#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputsMLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_conv2d_layer_call_and_return_conditional_losses_47043

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_3_layer_call_fn_48499

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_47711
conv2d_input
conv2d_1_input
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_1_inputconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_476722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input
??
?
__inference__traced_save_48844
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_2_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_4_bias_read_readvariableop8
4savev2_lstm_3_lstm_cell_5_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_5_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_4_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_5_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_5_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_4_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_5_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_5_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?#
value?#B?#BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?
value?B?BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_2_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_4_bias_read_readvariableop4savev2_lstm_3_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *P
dtypesF
D2B	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : :`0:0:	?::<H:H: : : : : :	0?:
??:?:	?:	<?:?: : : : : : : : : : : : :`0:0:	?::<H:H:	0?:
??:?:	?:	<?:?: : : : : : :`0:0:	?::<H:H:	0?:
??:?:	?:	<?:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:`0: 

_output_shapes
:0:%	!

_output_shapes
:	?: 


_output_shapes
::$ 

_output_shapes

:<H: 

_output_shapes
:H:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	0?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?:%!

_output_shapes
:	<?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
: :$$ 

_output_shapes

:`0: %

_output_shapes
:0:%&!

_output_shapes
:	?: '

_output_shapes
::$( 

_output_shapes

:<H: )

_output_shapes
:H:%*!

_output_shapes
:	0?:&+"
 
_output_shapes
:
??:!,

_output_shapes	
:?:%-!

_output_shapes
:	?:%.!

_output_shapes
:	<?:!/

_output_shapes	
:?:,0(
&
_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
: : 5

_output_shapes
: :$6 

_output_shapes

:`0: 7

_output_shapes
:0:%8!

_output_shapes
:	?: 9

_output_shapes
::$: 

_output_shapes

:<H: ;

_output_shapes
:H:%<!

_output_shapes
:	0?:&="
 
_output_shapes
:
??:!>

_output_shapes	
:?:%?!

_output_shapes
:	?:%@!

_output_shapes
:	<?:!A

_output_shapes	
:?:B

_output_shapes
: 
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_47090

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_47287

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48123

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?J
?
@__inference_model_layer_call_and_return_conditional_losses_47565

inputs
inputs_1
inputs_2
conv2d_1_47506
conv2d_1_47508
conv2d_47511
conv2d_47513
conv2d_2_47516
conv2d_2_47518
dense_4_47533
dense_4_47535
lstm_2_47540
lstm_2_47542
lstm_2_47544
dense_5_47547
dense_5_47549
lstm_3_47552
lstm_3_47554
lstm_3_47556
dense_6_47559
dense_6_47561
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_1_47506conv2d_1_47508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_470172"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_47511conv2d_47513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_470432 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_47516conv2d_2_47518*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_470692"
 conv2d_2/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_470902
activation_3/PartitionedCall?
activation_2/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_471032
activation_2/PartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_471162
activation_4/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_466552!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_466432
max_pooling2d/PartitionedCallu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMul&max_pooling2d/PartitionedCall:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul(max_pooling2d_1/PartitionedCall:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul%activation_4/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mul?
concatenate/PartitionedCallPartitionedCalltf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_471402
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_47533dense_4_47535*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_471612!
dense_4/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_466672!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_47540lstm_2_47542lstm_2_47544*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472122 
lstm_2/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_5_47547dense_5_47549*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_472872!
dense_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0lstm_3_47552lstm_3_47554lstm_3_47556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473382 
lstm_3/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_6_47559dense_6_47561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_474152!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?
@__inference_model_layer_call_and_return_conditional_losses_47432
conv2d_input
conv2d_1_input
conv2d_2_input
conv2d_1_47028
conv2d_1_47030
conv2d_47054
conv2d_47056
conv2d_2_47080
conv2d_2_47082
dense_4_47172
dense_4_47174
lstm_2_47269
lstm_2_47271
lstm_2_47273
dense_5_47298
dense_5_47300
lstm_3_47397
lstm_3_47399
lstm_3_47401
dense_6_47426
dense_6_47428
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_47028conv2d_1_47030*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_470172"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_47054conv2d_47056*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_470432 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_47080conv2d_2_47082*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_470692"
 conv2d_2/StatefulPartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_470902
activation_3/PartitionedCall?
activation_2/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_471032
activation_2/PartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_471162
activation_4/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_466552!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_466432
max_pooling2d/PartitionedCallu
tf.math.multiply/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply/Mul/y?
tf.math.multiply/MulMul&max_pooling2d/PartitionedCall:output:0tf.math.multiply/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply/Muly
tf.math.multiply_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_1/Mul/y?
tf.math.multiply_1/MulMul(max_pooling2d_1/PartitionedCall:output:0!tf.math.multiply_1/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_1/Muly
tf.math.multiply_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_2/Mul/y?
tf.math.multiply_2/MulMul%activation_4/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_2/Mul?
concatenate/PartitionedCallPartitionedCalltf.math.multiply/Mul:z:0tf.math.multiply_1/Mul:z:0tf.math.multiply_2/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_471402
concatenate/PartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_4_47172dense_4_47174*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_471612!
dense_4/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_466672!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_47269lstm_2_47271lstm_2_47273*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472122 
lstm_2/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_5_47298dense_5_47300*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_472872!
dense_5/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0lstm_3_47397lstm_3_47399lstm_3_47401*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_473382 
lstm_3/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_6_47426dense_6_47428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_474152!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:^ Z
0
_output_shapes
:??????????
&
_user_specified_nameconv2d_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_1_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_2_input
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48582
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2	
zeros_1l
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/hidden_sizel
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2	
MLCLSTMo
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
Reshape/shapey
ReshapeReshapeMLCLSTM:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2	
Reshape?
IdentityIdentityReshape:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
&__inference_lstm_2_layer_call_fn_48295

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_472122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
H
,__inference_activation_2_layer_call_fn_48142

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_471032
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48162

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48340
inputs_0#
mlclstm_readvariableop_resource%
!mlclstm_readvariableop_1_resource%
!mlclstm_readvariableop_2_resource
identity??MLCLSTM/ReadVariableOp?MLCLSTM/ReadVariableOp_1?MLCLSTM/ReadVariableOp_2F
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2	
zeros_1m
MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/hidden_sizem
MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
MLCLSTM/output_size?
MLCLSTM/ReadVariableOpReadVariableOpmlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
MLCLSTM/ReadVariableOp?
MLCLSTM/ReadVariableOp_1ReadVariableOp!mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
MLCLSTM/ReadVariableOp_1?
MLCLSTM/ReadVariableOp_2ReadVariableOp!mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
MLCLSTM/ReadVariableOp_2?
MLCLSTMMLCLSTMinputs_0MLCLSTM/hidden_size:output:0MLCLSTM/output_size:output:0MLCLSTM/ReadVariableOp:value:0 MLCLSTM/ReadVariableOp_1:value:0 MLCLSTM/ReadVariableOp_2:value:0*
T0*5
_output_shapes#
!:???????????????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
?
&__inference_lstm_2_layer_call_fn_48385
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_467792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????0
"
_user_specified_name
inputs/0
?
|
'__inference_dense_6_layer_call_fn_48624

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_474152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
?
|
'__inference_dense_5_layer_call_fn_48416

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_472872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?#
!__inference__traced_restore_49049
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias%
!assignvariableop_6_dense_4_kernel#
assignvariableop_7_dense_4_bias%
!assignvariableop_8_dense_5_kernel#
assignvariableop_9_dense_5_bias&
"assignvariableop_10_dense_6_kernel$
 assignvariableop_11_dense_6_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate1
-assignvariableop_17_lstm_2_lstm_cell_4_kernel;
7assignvariableop_18_lstm_2_lstm_cell_4_recurrent_kernel/
+assignvariableop_19_lstm_2_lstm_cell_4_bias1
-assignvariableop_20_lstm_3_lstm_cell_5_kernel;
7assignvariableop_21_lstm_3_lstm_cell_5_recurrent_kernel/
+assignvariableop_22_lstm_3_lstm_cell_5_bias
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1
assignvariableop_27_total_2
assignvariableop_28_count_2,
(assignvariableop_29_adam_conv2d_kernel_m*
&assignvariableop_30_adam_conv2d_bias_m.
*assignvariableop_31_adam_conv2d_1_kernel_m,
(assignvariableop_32_adam_conv2d_1_bias_m.
*assignvariableop_33_adam_conv2d_2_kernel_m,
(assignvariableop_34_adam_conv2d_2_bias_m-
)assignvariableop_35_adam_dense_4_kernel_m+
'assignvariableop_36_adam_dense_4_bias_m-
)assignvariableop_37_adam_dense_5_kernel_m+
'assignvariableop_38_adam_dense_5_bias_m-
)assignvariableop_39_adam_dense_6_kernel_m+
'assignvariableop_40_adam_dense_6_bias_m8
4assignvariableop_41_adam_lstm_2_lstm_cell_4_kernel_mB
>assignvariableop_42_adam_lstm_2_lstm_cell_4_recurrent_kernel_m6
2assignvariableop_43_adam_lstm_2_lstm_cell_4_bias_m8
4assignvariableop_44_adam_lstm_3_lstm_cell_5_kernel_mB
>assignvariableop_45_adam_lstm_3_lstm_cell_5_recurrent_kernel_m6
2assignvariableop_46_adam_lstm_3_lstm_cell_5_bias_m,
(assignvariableop_47_adam_conv2d_kernel_v*
&assignvariableop_48_adam_conv2d_bias_v.
*assignvariableop_49_adam_conv2d_1_kernel_v,
(assignvariableop_50_adam_conv2d_1_bias_v.
*assignvariableop_51_adam_conv2d_2_kernel_v,
(assignvariableop_52_adam_conv2d_2_bias_v-
)assignvariableop_53_adam_dense_4_kernel_v+
'assignvariableop_54_adam_dense_4_bias_v-
)assignvariableop_55_adam_dense_5_kernel_v+
'assignvariableop_56_adam_dense_5_bias_v-
)assignvariableop_57_adam_dense_6_kernel_v+
'assignvariableop_58_adam_dense_6_bias_v8
4assignvariableop_59_adam_lstm_2_lstm_cell_4_kernel_vB
>assignvariableop_60_adam_lstm_2_lstm_cell_4_recurrent_kernel_v6
2assignvariableop_61_adam_lstm_2_lstm_cell_4_bias_v8
4assignvariableop_62_adam_lstm_3_lstm_cell_5_kernel_vB
>assignvariableop_63_adam_lstm_3_lstm_cell_5_recurrent_kernel_v6
2assignvariableop_64_adam_lstm_3_lstm_cell_5_bias_v
identity_66??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?#
value?#B?#BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?
value?B?BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*P
dtypesF
D2B	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp-assignvariableop_17_lstm_2_lstm_cell_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp7assignvariableop_18_lstm_2_lstm_cell_4_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_lstm_2_lstm_cell_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_lstm_3_lstm_cell_5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp7assignvariableop_21_lstm_3_lstm_cell_5_recurrent_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_lstm_3_lstm_cell_5_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_2Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_lstm_2_lstm_cell_4_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_lstm_2_lstm_cell_4_recurrent_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_lstm_2_lstm_cell_4_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp4assignvariableop_44_adam_lstm_3_lstm_cell_5_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp>assignvariableop_45_adam_lstm_3_lstm_cell_5_recurrent_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp2assignvariableop_46_adam_lstm_3_lstm_cell_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv2d_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv2d_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_4_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_4_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_5_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_5_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_6_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_6_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_lstm_2_lstm_cell_4_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp>assignvariableop_60_adam_lstm_2_lstm_cell_4_recurrent_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_lstm_2_lstm_cell_4_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_lstm_3_lstm_cell_5_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adam_lstm_3_lstm_cell_5_recurrent_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_lstm_3_lstm_cell_5_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_649
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_65Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_65?
Identity_66IdentityIdentity_65:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_66"#
identity_66Identity_66:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
H
,__inference_activation_4_layer_call_fn_48181

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_471162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_46649

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_466432
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
R
conv2d_1_input@
 serving_default_conv2d_1_input:0??????????
R
conv2d_2_input@
 serving_default_conv2d_2_input:0??????????
N
conv2d_input>
serving_default_conv2d_input:0??????????;
dense_60
StatefulPartitionedCall:0?????????Htensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer_with_weights-3
layer-15
layer-16
layer-17
layer_with_weights-4
layer-18
layer_with_weights-5
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"ģ
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "name": "conv2d_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}, "name": "conv2d_1_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["conv2d_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d_1_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}, "name": "conv2d_2_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_2_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["max_pooling2d", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["max_pooling2d_1", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["activation_4", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["tf.math.multiply", 0, 0, {}], ["tf.math.multiply_1", 0, 0, {}], ["tf.math.multiply_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze", "inbound_nodes": [["max_pooling2d_2", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["tf.compat.v1.squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}], "input_layers": [["conv2d_input", 0, 0], ["conv2d_1_input", 0, 0], ["conv2d_2_input", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 144, 5]}, {"class_name": "TensorShape", "items": [null, 1, 144, 5]}, {"class_name": "TensorShape", "items": [null, 1, 144, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "name": "conv2d_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}, "name": "conv2d_1_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["conv2d_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d_1_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}, "name": "conv2d_2_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_2_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["max_pooling2d", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["max_pooling2d_1", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["activation_4", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["tf.math.multiply", 0, 0, {}], ["tf.math.multiply_1", 0, 0, {}], ["tf.math.multiply_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze", "inbound_nodes": [["max_pooling2d_2", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["tf.compat.v1.squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}], "input_layers": [["conv2d_input", 0, 0], ["conv2d_1_input", 0, 0], ["conv2d_2_input", 0, 0]], "output_layers": [["dense_6", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_1_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}
?


kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?


#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_2_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}
?
)trainable_variables
*	variables
+regularization_losses
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
-trainable_variables
.	variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


1kernel
2bias
3trainable_variables
4	variables
5regularization_losses
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;trainable_variables
<	variables
=regularization_losses
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
C	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
D	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
E	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 144, 32]}, {"class_name": "TensorShape", "items": [null, 1, 144, 32]}, {"class_name": "TensorShape", "items": [null, 1, 144, 32]}]}
?

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 96]}}
?
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
T	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.squeeze", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}}
?
Ucell
V
state_spec
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 48]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 48]}}
?

[kernel
\bias
]trainable_variables
^	variables
_regularization_losses
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 200]}}
?
acell
b
state_spec
ctrainable_variables
d	variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 30]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 30]}}
?

gkernel
hbias
itrainable_variables
j	variables
kregularization_losses
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?1m?2m?Jm?Km?[m?\m?gm?hm?rm?sm?tm?um?vm?wm?v?v?#v?$v?1v?2v?Jv?Kv?[v?\v?gv?hv?rv?sv?tv?uv?vv?wv?"
	optimizer
?
0
1
#2
$3
14
25
J6
K7
r8
s9
t10
[11
\12
u13
v14
w15
g16
h17"
trackable_list_wrapper
?
0
1
#2
$3
14
25
J6
K7
r8
s9
t10
[11
\12
u13
v14
w15
g16
h17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables
ymetrics
zlayer_regularization_losses
{layer_metrics
trainable_variables

|layers
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables
~metrics
layer_regularization_losses
?layer_metrics
trainable_variables
?layers
 	variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
%trainable_variables
?layers
&	variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
)trainable_variables
?layers
*	variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
-trainable_variables
?layers
.	variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_2/kernel
: 2conv2d_2/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
3trainable_variables
?layers
4	variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
7trainable_variables
?layers
8	variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
;trainable_variables
?layers
<	variables
=regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
@	variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ftrainable_variables
?layers
G	variables
Hregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :`02dense_4/kernel
:02dense_4/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ltrainable_variables
?layers
M	variables
Nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
Ptrainable_variables
?layers
Q	variables
Rregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?

rkernel
srecurrent_kernel
tbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_4", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?states
?metrics
 ?layer_regularization_losses
?layer_metrics
Wtrainable_variables
?layers
X	variables
Yregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_5/kernel
:2dense_5/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
]trainable_variables
?layers
^	variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

ukernel
vrecurrent_kernel
wbias
?trainable_variables
?	variables
?regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_5", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?states
?metrics
 ?layer_regularization_losses
?layer_metrics
ctrainable_variables
?layers
d	variables
eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :<H2dense_6/kernel
:H2dense_6/bias
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
itrainable_variables
?layers
j	variables
kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	0?2lstm_2/lstm_cell_4/kernel
7:5
??2#lstm_2/lstm_cell_4/recurrent_kernel
&:$?2lstm_2/lstm_cell_4/bias
,:*	?2lstm_3/lstm_cell_5/kernel
6:4	<?2#lstm_3/lstm_cell_5/recurrent_kernel
&:$?2lstm_3/lstm_cell_5/bias
 "
trackable_list_wrapper
8
?0
?1
?2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
5
r0
s1
t2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
U0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
5
u0
v1
w2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?layer_metrics
?trainable_variables
?layers
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, 2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
.:, 2Adam/conv2d_2/kernel/m
 : 2Adam/conv2d_2/bias/m
%:#`02Adam/dense_4/kernel/m
:02Adam/dense_4/bias/m
&:$	?2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
%:#<H2Adam/dense_6/kernel/m
:H2Adam/dense_6/bias/m
1:/	0?2 Adam/lstm_2/lstm_cell_4/kernel/m
<::
??2*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m
+:)?2Adam/lstm_2/lstm_cell_4/bias/m
1:/	?2 Adam/lstm_3/lstm_cell_5/kernel/m
;:9	<?2*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m
+:)?2Adam/lstm_3/lstm_cell_5/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, 2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
.:, 2Adam/conv2d_2/kernel/v
 : 2Adam/conv2d_2/bias/v
%:#`02Adam/dense_4/kernel/v
:02Adam/dense_4/bias/v
&:$	?2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
%:#<H2Adam/dense_6/kernel/v
:H2Adam/dense_6/bias/v
1:/	0?2 Adam/lstm_2/lstm_cell_4/kernel/v
<::
??2*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v
+:)?2Adam/lstm_2/lstm_cell_4/bias/v
1:/	?2 Adam/lstm_3/lstm_cell_5/kernel/v
;:9	<?2*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v
+:)?2Adam/lstm_3/lstm_cell_5/bias/v
?2?
@__inference_model_layer_call_and_return_conditional_losses_47886
@__inference_model_layer_call_and_return_conditional_losses_48008
@__inference_model_layer_call_and_return_conditional_losses_47496
@__inference_model_layer_call_and_return_conditional_losses_47432?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_model_layer_call_fn_48051
%__inference_model_layer_call_fn_48094
%__inference_model_layer_call_fn_47711
%__inference_model_layer_call_fn_47604?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_46637?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_48104?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_48113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_48132?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_2_layer_call_and_return_conditional_losses_48137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_2_layer_call_fn_48142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_3_layer_call_and_return_conditional_losses_48147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_3_layer_call_fn_48152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_2_layer_call_fn_48171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_46649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46655?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_1_layer_call_fn_46661?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_activation_4_layer_call_and_return_conditional_losses_48176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_activation_4_layer_call_fn_48181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_48189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_48196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_48207?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_48216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_46667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_2_layer_call_fn_46673?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48374
A__inference_lstm_2_layer_call_and_return_conditional_losses_48340
A__inference_lstm_2_layer_call_and_return_conditional_losses_48284
A__inference_lstm_2_layer_call_and_return_conditional_losses_48250?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_2_layer_call_fn_48385
&__inference_lstm_2_layer_call_fn_48396
&__inference_lstm_2_layer_call_fn_48306
&__inference_lstm_2_layer_call_fn_48295?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_48407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_48416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48582
A__inference_lstm_3_layer_call_and_return_conditional_losses_48546
A__inference_lstm_3_layer_call_and_return_conditional_losses_48488
A__inference_lstm_3_layer_call_and_return_conditional_losses_48452?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_lstm_3_layer_call_fn_48499
&__inference_lstm_3_layer_call_fn_48510
&__inference_lstm_3_layer_call_fn_48593
&__inference_lstm_3_layer_call_fn_48604?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dense_6_layer_call_and_return_conditional_losses_48615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_6_layer_call_fn_48624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_47764conv2d_1_inputconv2d_2_inputconv2d_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_46637?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
? "1?.
,
dense_6!?
dense_6?????????H?
G__inference_activation_2_layer_call_and_return_conditional_losses_48137j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_2_layer_call_fn_48142]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
G__inference_activation_3_layer_call_and_return_conditional_losses_48147j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_3_layer_call_fn_48152]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
G__inference_activation_4_layer_call_and_return_conditional_losses_48176j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_4_layer_call_fn_48181]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
F__inference_concatenate_layer_call_and_return_conditional_losses_48189????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? ".?+
$?!
0??????????`
? ?
+__inference_concatenate_layer_call_fn_48196????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? "!???????????`?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_48123n#$8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_1_layer_call_fn_48132a#$8?5
.?+
)?&
inputs??????????
? "!??????????? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_48162n128?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_2_layer_call_fn_48171a128?5
.?+
)?&
inputs??????????
? "!??????????? ?
A__inference_conv2d_layer_call_and_return_conditional_losses_48104n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
&__inference_conv2d_layer_call_fn_48113a8?5
.?+
)?&
inputs??????????
? "!??????????? ?
B__inference_dense_4_layer_call_and_return_conditional_losses_48207nJK8?5
.?+
)?&
inputs??????????`
? ".?+
$?!
0??????????0
? ?
'__inference_dense_4_layer_call_fn_48216aJK8?5
.?+
)?&
inputs??????????`
? "!???????????0?
B__inference_dense_5_layer_call_and_return_conditional_losses_48407g[\5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????
? ?
'__inference_dense_5_layer_call_fn_48416Z[\5?2
+?(
&?#
inputs???????????
? "????????????
B__inference_dense_6_layer_call_and_return_conditional_losses_48615\gh/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????H
? z
'__inference_dense_6_layer_call_fn_48624Ogh/?,
%?"
 ?
inputs?????????<
? "??????????H?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48250trst@?=
6?3
%?"
inputs??????????0

 
p

 
? "+?(
!?
0???????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48284trst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "+?(
!?
0???????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48340?rstO?L
E?B
4?1
/?,
inputs/0??????????????????0

 
p

 
? "3?0
)?&
0???????????????????
? ?
A__inference_lstm_2_layer_call_and_return_conditional_losses_48374?rstO?L
E?B
4?1
/?,
inputs/0??????????????????0

 
p 

 
? "3?0
)?&
0???????????????????
? ?
&__inference_lstm_2_layer_call_fn_48295grst@?=
6?3
%?"
inputs??????????0

 
p

 
? "?????????????
&__inference_lstm_2_layer_call_fn_48306grst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "?????????????
&__inference_lstm_2_layer_call_fn_48385~rstO?L
E?B
4?1
/?,
inputs/0??????????????????0

 
p

 
? "&?#????????????????????
&__inference_lstm_2_layer_call_fn_48396~rstO?L
E?B
4?1
/?,
inputs/0??????????????????0

 
p 

 
? "&?#????????????????????
A__inference_lstm_3_layer_call_and_return_conditional_losses_48452nuvw@?=
6?3
%?"
inputs??????????

 
p

 
? "%?"
?
0?????????<
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48488nuvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "%?"
?
0?????????<
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48546}uvwO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????<
? ?
A__inference_lstm_3_layer_call_and_return_conditional_losses_48582}uvwO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????<
? ?
&__inference_lstm_3_layer_call_fn_48499auvw@?=
6?3
%?"
inputs??????????

 
p

 
? "??????????<?
&__inference_lstm_3_layer_call_fn_48510auvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "??????????<?
&__inference_lstm_3_layer_call_fn_48593puvwO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "??????????<?
&__inference_lstm_3_layer_call_fn_48604puvwO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "??????????<?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_46655?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_46661?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_46667?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_46673?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_46643?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_46649?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
@__inference_model_layer_call_and_return_conditional_losses_47432?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
p

 
? "%?"
?
0?????????H
? ?
@__inference_model_layer_call_and_return_conditional_losses_47496?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
p 

 
? "%?"
?
0?????????H
? ?
@__inference_model_layer_call_and_return_conditional_losses_47886?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p

 
? "%?"
?
0?????????H
? ?
@__inference_model_layer_call_and_return_conditional_losses_48008?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p 

 
? "%?"
?
0?????????H
? ?
%__inference_model_layer_call_fn_47604?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
p

 
? "??????????H?
%__inference_model_layer_call_fn_47711?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
p 

 
? "??????????H?
%__inference_model_layer_call_fn_48051?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p

 
? "??????????H?
%__inference_model_layer_call_fn_48094?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p 

 
? "??????????H?
#__inference_signature_wrapper_47764?#$12JKrst[\uvwgh???
? 
???
C
conv2d_1_input1?.
conv2d_1_input??????????
C
conv2d_2_input1?.
conv2d_2_input??????????
?
conv2d_input/?,
conv2d_input??????????"1?.
,
dense_6!?
dense_6?????????H