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
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:`0*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:0*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	<?*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:?*
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
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:`0*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_4/kernel/m
?
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*&
shared_nameAdam/dense_5/kernel/m
?
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	<?*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:?*
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
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:`0*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_4/kernel/v
?
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*&
shared_nameAdam/dense_5/kernel/v
?
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	<?*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes	
:?*
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
?r
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?r
value?rB?q B?q
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
 
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
 
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
R
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
R
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api

C	keras_api

D	keras_api

E	keras_api
R
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api

T	keras_api
l
Ucell
V
state_spec
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
h

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
l
acell
b
state_spec
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

gkernel
hbias
i	variables
jtrainable_variables
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
xlayer_metrics
	variables
trainable_variables
ymetrics
regularization_losses

zlayers
{non_trainable_variables
|layer_regularization_losses
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
}layer_metrics
	variables
 trainable_variables
~metrics
!regularization_losses

layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
%	variables
&trainable_variables
?metrics
'regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
)	variables
*trainable_variables
?metrics
+regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
-	variables
.trainable_variables
?metrics
/regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
3	variables
4trainable_variables
?metrics
5regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
7	variables
8trainable_variables
?metrics
9regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
;	variables
<trainable_variables
?metrics
=regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?	variables
@trainable_variables
?metrics
Aregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
 
 
 
?
?layer_metrics
F	variables
Gtrainable_variables
?metrics
Hregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
?layer_metrics
L	variables
Mtrainable_variables
?metrics
Nregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
P	variables
Qtrainable_variables
?metrics
Rregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
 
?

rkernel
srecurrent_kernel
tbias
?	variables
?trainable_variables
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
?layer_metrics
W	variables
?states
Xtrainable_variables
?metrics
Yregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
?
?layer_metrics
]	variables
^trainable_variables
?metrics
_regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?

ukernel
vrecurrent_kernel
wbias
?	variables
?trainable_variables
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
?layer_metrics
c	variables
?states
dtrainable_variables
?metrics
eregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1

g0
h1
 
?
?layer_metrics
i	variables
jtrainable_variables
?metrics
kregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
US
VARIABLE_VALUElstm_2/lstm_cell_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#lstm_2/lstm_cell_4/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_2/lstm_cell_4/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_3/lstm_cell_5/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#lstm_3/lstm_cell_5/recurrent_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUElstm_3/lstm_cell_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
?2
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
?layer_metrics
?	variables
?trainable_variables
?metrics
?regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
?	variables
?trainable_variables
?metrics
?regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_1_inputserving_default_conv2d_2_inputserving_default_conv2d_inputconv2d_1/kernelconv2d_1/biasconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasdense_3/kerneldense_3/biaslstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biasdense_4/kerneldense_4/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biasdense_5/kerneldense_5/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_33853
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_2/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_4/bias/Read/ReadVariableOp-lstm_3/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/v/Read/ReadVariableOpConst*N
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
__inference__traced_save_34933
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biastotalcounttotal_1count_1total_2count_2Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/lstm_2/lstm_cell_4/kernel/m*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mAdam/lstm_2/lstm_cell_4/bias/m Adam/lstm_3/lstm_cell_5/kernel/m*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mAdam/lstm_3/lstm_cell_5/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/lstm_2/lstm_cell_4/kernel/v*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vAdam/lstm_2/lstm_cell_4/bias/v Adam/lstm_3/lstm_cell_5/kernel/v*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vAdam/lstm_3/lstm_cell_5/bias/v*M
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
!__inference__traced_restore_35138??
?
|
'__inference_dense_5_layer_call_fn_34713

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
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_335042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????<::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
??
?
__inference__traced_save_34933
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
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
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop?
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
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop?
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
ShardedFilename?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?!
value?!B?!BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?
value?B?BB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_2_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_4_bias_read_readvariableop4savev2_lstm_3_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : : : : : : :`0:0:	?::	<?:?: : : : : :	0?:
??:?:	?:	<?:?: : : : : : : : : : : : :`0:0:	?::	<?:?:	0?:
??:?:	?:	<?:?: : : : : : :`0:0:	?::	<?:?:	0?:
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
::%!

_output_shapes
:	<?:!

_output_shapes	
:?:
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
::%(!

_output_shapes
:	<?:!)

_output_shapes	
:?:%*!

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
::%:!

_output_shapes
:	<?:!;

_output_shapes	
:?:%<!

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
a
E__inference_activation_layer_call_and_return_conditional_losses_33192

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
F
*__inference_activation_layer_call_fn_34231

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
GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_331922
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
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_34577
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
ؐ
?#
!__inference__traced_restore_35138
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias%
!assignvariableop_6_dense_3_kernel#
assignvariableop_7_dense_3_bias%
!assignvariableop_8_dense_4_kernel#
assignvariableop_9_dense_4_bias&
"assignvariableop_10_dense_5_kernel$
 assignvariableop_11_dense_5_bias!
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
)assignvariableop_35_adam_dense_3_kernel_m+
'assignvariableop_36_adam_dense_3_bias_m-
)assignvariableop_37_adam_dense_4_kernel_m+
'assignvariableop_38_adam_dense_4_bias_m-
)assignvariableop_39_adam_dense_5_kernel_m+
'assignvariableop_40_adam_dense_5_bias_m8
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
)assignvariableop_53_adam_dense_3_kernel_v+
'assignvariableop_54_adam_dense_3_bias_v-
)assignvariableop_55_adam_dense_4_kernel_v+
'assignvariableop_56_adam_dense_4_bias_v-
)assignvariableop_57_adam_dense_5_kernel_v+
'assignvariableop_58_adam_dense_5_bias_v8
4assignvariableop_59_adam_lstm_2_lstm_cell_4_kernel_vB
>assignvariableop_60_adam_lstm_2_lstm_cell_4_recurrent_kernel_v6
2assignvariableop_61_adam_lstm_2_lstm_cell_4_bias_v8
4assignvariableop_62_adam_lstm_3_lstm_cell_5_kernel_vB
>assignvariableop_63_adam_lstm_3_lstm_cell_5_recurrent_kernel_v6
2assignvariableop_64_adam_lstm_3_lstm_cell_5_bias_v
identity_66??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:B*
dtype0*?!
value?!B?!BB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_5_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_4_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_4_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_5_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_5_bias_mIdentity_40:output:0"/device:CPU:0*
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
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_4_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_4_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_5_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_5_bias_vIdentity_58:output:0"/device:CPU:0*
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
?
~
F__inference_concatenate_layer_call_and_return_conditional_losses_33229

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
?K
?
@__inference_model_layer_call_and_return_conditional_losses_33585
conv2d_input
conv2d_1_input
conv2d_2_input
conv2d_1_33526
conv2d_1_33528
conv2d_33531
conv2d_33533
conv2d_2_33536
conv2d_2_33538
dense_3_33553
dense_3_33555
lstm_2_33560
lstm_2_33562
lstm_2_33564
dense_4_33567
dense_4_33569
lstm_3_33572
lstm_3_33574
lstm_3_33576
dense_5_33579
dense_5_33581
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_33526conv2d_1_33528*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_331062"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_33531conv2d_33533*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_331322 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_33536conv2d_2_33538*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_331582"
 conv2d_2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
G__inference_activation_1_layer_call_and_return_conditional_losses_331792
activation_1/PartitionedCall?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_331922
activation/PartitionedCall?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_2_layer_call_and_return_conditional_losses_332052
activation_2/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_327442!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_327322
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
tf.math.multiply_2/MulMul%activation_2/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_332292
concatenate/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_33553dense_3_33555*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_332502!
dense_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_327562!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_33560lstm_2_33562lstm_2_33564*
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333352 
lstm_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_33567dense_4_33569*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_333762!
dense_4/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0lstm_3_33572lstm_3_33574lstm_3_33576*
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334632 
lstm_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_5_33579dense_5_33581*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_335042!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_34671

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
H
,__inference_activation_2_layer_call_fn_34270

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
G__inference_activation_2_layer_call_and_return_conditional_losses_332052
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
?
c
G__inference_activation_1_layer_call_and_return_conditional_losses_34236

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
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_34265

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
I
-__inference_max_pooling2d_layer_call_fn_32738

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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_327322
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
?
?
%__inference_model_layer_call_fn_33800
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
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_337612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?

?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_33106

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

?
B__inference_dense_3_layer_call_and_return_conditional_losses_33250

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
?
a
E__inference_activation_layer_call_and_return_conditional_losses_34226

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
?
e
+__inference_concatenate_layer_call_fn_34285
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
F__inference_concatenate_layer_call_and_return_conditional_losses_332292
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
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_33034

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
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_34704

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
??
?
 __inference__wrapped_model_32726
conv2d_input
conv2d_1_input
conv2d_2_input1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource3
/model_dense_3_mlcmatmul_readvariableop_resource1
-model_dense_3_biasadd_readvariableop_resource0
,model_lstm_2_mlclstm_readvariableop_resource2
.model_lstm_2_mlclstm_readvariableop_1_resource2
.model_lstm_2_mlclstm_readvariableop_2_resource3
/model_dense_4_mlcmatmul_readvariableop_resource1
-model_dense_4_biasadd_readvariableop_resource0
,model_lstm_3_mlclstm_readvariableop_resource2
.model_lstm_3_mlclstm_readvariableop_1_resource2
.model_lstm_3_mlclstm_readvariableop_2_resource3
/model_dense_5_mlcmatmul_readvariableop_resource1
-model_dense_5_biasadd_readvariableop_resource
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?$model/dense_3/BiasAdd/ReadVariableOp?&model/dense_3/MLCMatMul/ReadVariableOp?$model/dense_4/BiasAdd/ReadVariableOp?&model/dense_4/MLCMatMul/ReadVariableOp?$model/dense_5/BiasAdd/ReadVariableOp?&model/dense_5/MLCMatMul/ReadVariableOp?#model/lstm_2/MLCLSTM/ReadVariableOp?%model/lstm_2/MLCLSTM/ReadVariableOp_1?%model/lstm_2/MLCLSTM/ReadVariableOp_2?#model/lstm_3/MLCLSTM/ReadVariableOp?%model/lstm_3/MLCLSTM/ReadVariableOp_1?%model/lstm_3/MLCLSTM/ReadVariableOp_2?
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
model/activation_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation_1/Relu?
model/activation/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation/Relu?
model/activation_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model/activation_2/Relu?
model/max_pooling2d_1/MaxPoolMaxPool%model/activation_1/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/max_pooling2d/MaxPoolMaxPool#model/activation/Relu:activations:0*0
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
model/tf.math.multiply_2/MulMul%model/activation_2/Relu:activations:0'model/tf.math.multiply_2/Mul/y:output:0*
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
&model/dense_3/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_3_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02(
&model/dense_3/MLCMatMul/ReadVariableOp?
model/dense_3/MLCMatMul	MLCMatMul!model/concatenate/concat:output:0.model/dense_3/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
model/dense_3/MLCMatMul?
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model/dense_3/BiasAdd/ReadVariableOp?
model/dense_3/BiasAddBiasAdd!model/dense_3/MLCMatMul:product:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
model/dense_3/BiasAdd?
model/dense_3/TanhTanhmodel/dense_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
model/dense_3/Tanh?
model/max_pooling2d_2/MaxPoolMaxPoolmodel/dense_3/Tanh:y:0*0
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
&model/dense_4/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_4_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model/dense_4/MLCMatMul/ReadVariableOp?
model/dense_4/MLCMatMul	MLCMatMulmodel/lstm_2/MLCLSTM:output:0.model/dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
model/dense_4/MLCMatMul?
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_4/BiasAdd/ReadVariableOp?
model/dense_4/BiasAddBiasAdd!model/dense_4/MLCMatMul:product:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model/dense_4/BiasAdd?
model/dense_4/TanhTanhmodel/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model/dense_4/Tanhn
model/lstm_3/ShapeShapemodel/dense_4/Tanh:y:0*
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
model/lstm_3/MLCLSTMMLCLSTMmodel/dense_4/Tanh:y:0)model/lstm_3/MLCLSTM/hidden_size:output:0)model/lstm_3/MLCLSTM/output_size:output:0+model/lstm_3/MLCLSTM/ReadVariableOp:value:0-model/lstm_3/MLCLSTM/ReadVariableOp_1:value:0-model/lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
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
&model/dense_5/MLCMatMul/ReadVariableOpReadVariableOp/model_dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02(
&model/dense_5/MLCMatMul/ReadVariableOp?
model/dense_5/MLCMatMul	MLCMatMulmodel/lstm_3/Reshape:output:0.model/dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_5/MLCMatMul?
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$model/dense_5/BiasAdd/ReadVariableOp?
model/dense_5/BiasAddBiasAdd!model/dense_5/MLCMatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model/dense_5/BiasAdd?
model/dense_5/TanhTanhmodel/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model/dense_5/Tanh?
IdentityIdentitymodel/dense_5/Tanh:y:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp'^model/dense_3/MLCMatMul/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/MLCMatMul/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp'^model/dense_5/MLCMatMul/ReadVariableOp$^model/lstm_2/MLCLSTM/ReadVariableOp&^model/lstm_2/MLCLSTM/ReadVariableOp_1&^model/lstm_2/MLCLSTM/ReadVariableOp_2$^model/lstm_3/MLCLSTM/ReadVariableOp&^model/lstm_3/MLCLSTM/ReadVariableOp_1&^model/lstm_3/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

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
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2P
&model/dense_3/MLCMatMul/ReadVariableOp&model/dense_3/MLCMatMul/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/MLCMatMul/ReadVariableOp&model/dense_4/MLCMatMul/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2P
&model/dense_5/MLCMatMul/ReadVariableOp&model/dense_5/MLCMatMul/ReadVariableOp2J
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
?J
?
@__inference_model_layer_call_and_return_conditional_losses_33654

inputs
inputs_1
inputs_2
conv2d_1_33595
conv2d_1_33597
conv2d_33600
conv2d_33602
conv2d_2_33605
conv2d_2_33607
dense_3_33622
dense_3_33624
lstm_2_33629
lstm_2_33631
lstm_2_33633
dense_4_33636
dense_4_33638
lstm_3_33641
lstm_3_33643
lstm_3_33645
dense_5_33648
dense_5_33650
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_1_33595conv2d_1_33597*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_331062"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_33600conv2d_33602*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_331322 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_33605conv2d_2_33607*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_331582"
 conv2d_2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
G__inference_activation_1_layer_call_and_return_conditional_losses_331792
activation_1/PartitionedCall?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_331922
activation/PartitionedCall?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_2_layer_call_and_return_conditional_losses_332052
activation_2/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_327442!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_327322
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
tf.math.multiply_2/MulMul%activation_2/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_332292
concatenate/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_33622dense_3_33624*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_332502!
dense_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_327562!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_33629lstm_2_33631lstm_2_33633*
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333012 
lstm_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_33636dense_4_33638*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_333762!
dense_4/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0lstm_3_33641lstm_3_33643lstm_3_33645*
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334272 
lstm_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_5_33648dense_5_33650*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_335042!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
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
?
%__inference_model_layer_call_fn_34140
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
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_336542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?J
?
@__inference_model_layer_call_and_return_conditional_losses_33761

inputs
inputs_1
inputs_2
conv2d_1_33702
conv2d_1_33704
conv2d_33707
conv2d_33709
conv2d_2_33712
conv2d_2_33714
dense_3_33729
dense_3_33731
lstm_2_33736
lstm_2_33738
lstm_2_33740
dense_4_33743
dense_4_33745
lstm_3_33748
lstm_3_33750
lstm_3_33752
dense_5_33755
dense_5_33757
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_1_33702conv2d_1_33704*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_331062"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_33707conv2d_33709*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_331322 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_33712conv2d_2_33714*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_331582"
 conv2d_2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
G__inference_activation_1_layer_call_and_return_conditional_losses_331792
activation_1/PartitionedCall?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_331922
activation/PartitionedCall?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_2_layer_call_and_return_conditional_losses_332052
activation_2/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_327442!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_327322
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
tf.math.multiply_2/MulMul%activation_2/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_332292
concatenate/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_33729dense_3_33731*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_332502!
dense_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_327562!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_33736lstm_2_33738lstm_2_33740*
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333352 
lstm_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_33743dense_4_33745*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_333762!
dense_4/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0lstm_3_33748lstm_3_33750lstm_3_33752*
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334632 
lstm_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_5_33755dense_5_33757*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_335042!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
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
?
%__inference_model_layer_call_fn_33693
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
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_336542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?K
?
@__inference_model_layer_call_and_return_conditional_losses_33521
conv2d_input
conv2d_1_input
conv2d_2_input
conv2d_1_33117
conv2d_1_33119
conv2d_33143
conv2d_33145
conv2d_2_33169
conv2d_2_33171
dense_3_33261
dense_3_33263
lstm_2_33358
lstm_2_33360
lstm_2_33362
dense_4_33387
dense_4_33389
lstm_3_33486
lstm_3_33488
lstm_3_33490
dense_5_33515
dense_5_33517
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?lstm_2/StatefulPartitionedCall?lstm_3/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_33117conv2d_1_33119*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_331062"
 conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_33143conv2d_33145*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_331322 
conv2d/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_33169conv2d_2_33171*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_331582"
 conv2d_2/StatefulPartitionedCall?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
G__inference_activation_1_layer_call_and_return_conditional_losses_331792
activation_1/PartitionedCall?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_331922
activation/PartitionedCall?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_2_layer_call_and_return_conditional_losses_332052
activation_2/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_327442!
max_pooling2d_1/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_327322
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
tf.math.multiply_2/MulMul%activation_2/PartitionedCall:output:0!tf.math.multiply_2/Mul/y:output:0*
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
F__inference_concatenate_layer_call_and_return_conditional_losses_332292
concatenate/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_3_33261dense_3_33263*
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
B__inference_dense_3_layer_call_and_return_conditional_losses_332502!
dense_3/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_327562!
max_pooling2d_2/PartitionedCall?
tf.compat.v1.squeeze/SqueezeSqueeze(max_pooling2d_2/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2
tf.compat.v1.squeeze/Squeeze?
lstm_2/StatefulPartitionedCallStatefulPartitionedCall%tf.compat.v1.squeeze/Squeeze:output:0lstm_2_33358lstm_2_33360lstm_2_33362*
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333012 
lstm_2/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0dense_4_33387dense_4_33389*
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
B__inference_dense_4_layer_call_and_return_conditional_losses_333762!
dense_4/StatefulPartitionedCall?
lstm_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0lstm_3_33486lstm_3_33488lstm_3_33490*
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334272 
lstm_3/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0dense_5_33515dense_5_33517*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_335042!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
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

?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_34251

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
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_34463

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
?
&__inference_lstm_3_layer_call_fn_34682

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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334272
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
?
|
'__inference_dense_3_layer_call_fn_34305

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
B__inference_dense_3_layer_call_and_return_conditional_losses_332502
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
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_34429

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
{
&__inference_conv2d_layer_call_fn_34202

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
A__inference_conv2d_layer_call_and_return_conditional_losses_331322
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_34212

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
?
c
G__inference_activation_2_layer_call_and_return_conditional_losses_33205

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
A__inference_lstm_2_layer_call_and_return_conditional_losses_33301

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
?
?
%__inference_model_layer_call_fn_34183
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
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_337612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?
?
&__inference_lstm_2_layer_call_fn_34485

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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333352
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
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_34496

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_33158

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
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_34296

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
??
?
@__inference_model_layer_call_and_return_conditional_losses_33975
inputs_0
inputs_1
inputs_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)dense_3_mlcmatmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&lstm_2_mlclstm_readvariableop_resource,
(lstm_2_mlclstm_readvariableop_1_resource,
(lstm_2_mlclstm_readvariableop_2_resource-
)dense_4_mlcmatmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&lstm_3_mlclstm_readvariableop_resource,
(lstm_3_mlclstm_readvariableop_1_resource,
(lstm_3_mlclstm_readvariableop_2_resource-
)dense_5_mlcmatmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/MLCMatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/MLCMatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp? dense_5/MLCMatMul/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp_1?lstm_2/MLCLSTM/ReadVariableOp_2?lstm_3/MLCLSTM/ReadVariableOp?lstm_3/MLCLSTM/ReadVariableOp_1?lstm_3/MLCLSTM/ReadVariableOp_2?
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
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_1/Relu~
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation/Relu?
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*0
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
tf.math.multiply_2/MulMulactivation_2/Relu:activations:0!tf.math.multiply_2/Mul/y:output:0*
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
 dense_3/MLCMatMul/ReadVariableOpReadVariableOp)dense_3_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_3/MLCMatMul/ReadVariableOp?
dense_3/MLCMatMul	MLCMatMulconcatenate/concat:output:0(dense_3/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_3/MLCMatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MLCMatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_3/BiasAddy
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_3/Tanh?
max_pooling2d_2/MaxPoolMaxPooldense_3/Tanh:y:0*0
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
 dense_4/MLCMatMul/ReadVariableOpReadVariableOp)dense_4_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_4/MLCMatMul/ReadVariableOp?
dense_4/MLCMatMul	MLCMatMullstm_2/MLCLSTM:output:0(dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_4/MLCMatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MLCMatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_4/BiasAddu
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_4/Tanh\
lstm_3/ShapeShapedense_4/Tanh:y:0*
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
lstm_3/MLCLSTMMLCLSTMdense_4/Tanh:y:0#lstm_3/MLCLSTM/hidden_size:output:0#lstm_3/MLCLSTM/output_size:output:0%lstm_3/MLCLSTM/ReadVariableOp:value:0'lstm_3/MLCLSTM/ReadVariableOp_1:value:0'lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
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
 dense_5/MLCMatMul/ReadVariableOpReadVariableOp)dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02"
 dense_5/MLCMatMul/ReadVariableOp?
dense_5/MLCMatMul	MLCMatMullstm_3/Reshape:output:0(dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MLCMatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MLCMatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Tanh?
IdentityIdentitydense_5/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/MLCMatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/MLCMatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/MLCMatMul/ReadVariableOp^lstm_2/MLCLSTM/ReadVariableOp ^lstm_2/MLCLSTM/ReadVariableOp_1 ^lstm_2/MLCLSTM/ReadVariableOp_2^lstm_3/MLCLSTM/ReadVariableOp ^lstm_3/MLCLSTM/ReadVariableOp_1 ^lstm_3/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

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
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/MLCMatMul/ReadVariableOp dense_3/MLCMatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/MLCMatMul/ReadVariableOp dense_4/MLCMatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/MLCMatMul/ReadVariableOp dense_5/MLCMatMul/ReadVariableOp2>
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
?
|
'__inference_dense_4_layer_call_fn_34505

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
B__inference_dense_4_layer_call_and_return_conditional_losses_333762
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
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_32732

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
?
?
&__inference_lstm_3_layer_call_fn_34599
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_330812
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
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_33463

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
A__inference_conv2d_layer_call_and_return_conditional_losses_33132

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
?
#__inference_signature_wrapper_33853
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
 *(
_output_shapes
:??????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_327262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

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
?
H
,__inference_activation_1_layer_call_fn_34241

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
G__inference_activation_1_layer_call_and_return_conditional_losses_331792
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
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_34339
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
?
K
/__inference_max_pooling2d_1_layer_call_fn_32750

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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_327442
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
?
?
&__inference_lstm_3_layer_call_fn_34588
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_330342
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
?
?
&__inference_lstm_3_layer_call_fn_34693

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
A__inference_lstm_3_layer_call_and_return_conditional_losses_334632
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
?"
?
A__inference_lstm_3_layer_call_and_return_conditional_losses_34635

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
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_34373
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_33081

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
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_32868

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
?
?
&__inference_lstm_2_layer_call_fn_34395
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_329132
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
G__inference_activation_1_layer_call_and_return_conditional_losses_33179

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
}
(__inference_conv2d_1_layer_call_fn_34221

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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_331062
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
B__inference_dense_4_layer_call_and_return_conditional_losses_33376

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
??
?
@__inference_model_layer_call_and_return_conditional_losses_34097
inputs_0
inputs_1
inputs_2+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource-
)dense_3_mlcmatmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&lstm_2_mlclstm_readvariableop_resource,
(lstm_2_mlclstm_readvariableop_1_resource,
(lstm_2_mlclstm_readvariableop_2_resource-
)dense_4_mlcmatmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&lstm_3_mlclstm_readvariableop_resource,
(lstm_3_mlclstm_readvariableop_1_resource,
(lstm_3_mlclstm_readvariableop_2_resource-
)dense_5_mlcmatmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/MLCMatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp? dense_4/MLCMatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp? dense_5/MLCMatMul/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp?lstm_2/MLCLSTM/ReadVariableOp_1?lstm_2/MLCLSTM/ReadVariableOp_2?lstm_3/MLCLSTM/ReadVariableOp?lstm_3/MLCLSTM/ReadVariableOp_1?lstm_3/MLCLSTM/ReadVariableOp_2?
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
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_1/Relu~
activation/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation/Relu?
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_2/Relu?
max_pooling2d_1/MaxPoolMaxPoolactivation_1/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
max_pooling2d/MaxPoolMaxPoolactivation/Relu:activations:0*0
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
tf.math.multiply_2/MulMulactivation_2/Relu:activations:0!tf.math.multiply_2/Mul/y:output:0*
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
 dense_3/MLCMatMul/ReadVariableOpReadVariableOp)dense_3_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_3/MLCMatMul/ReadVariableOp?
dense_3/MLCMatMul	MLCMatMulconcatenate/concat:output:0(dense_3/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_3/MLCMatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MLCMatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_3/BiasAddy
dense_3/TanhTanhdense_3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_3/Tanh?
max_pooling2d_2/MaxPoolMaxPooldense_3/Tanh:y:0*0
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
 dense_4/MLCMatMul/ReadVariableOpReadVariableOp)dense_4_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_4/MLCMatMul/ReadVariableOp?
dense_4/MLCMatMul	MLCMatMullstm_2/MLCLSTM:output:0(dense_4/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_4/MLCMatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MLCMatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_4/BiasAddu
dense_4/TanhTanhdense_4/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_4/Tanh\
lstm_3/ShapeShapedense_4/Tanh:y:0*
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
lstm_3/MLCLSTMMLCLSTMdense_4/Tanh:y:0#lstm_3/MLCLSTM/hidden_size:output:0#lstm_3/MLCLSTM/output_size:output:0%lstm_3/MLCLSTM/ReadVariableOp:value:0'lstm_3/MLCLSTM/ReadVariableOp_1:value:0'lstm_3/MLCLSTM/ReadVariableOp_2:value:0*
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
 dense_5/MLCMatMul/ReadVariableOpReadVariableOp)dense_5_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02"
 dense_5/MLCMatMul/ReadVariableOp?
dense_5/MLCMatMul	MLCMatMullstm_3/Reshape:output:0(dense_5/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/MLCMatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MLCMatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_5/BiasAddq
dense_5/TanhTanhdense_5/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_5/Tanh?
IdentityIdentitydense_5/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/MLCMatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/MLCMatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/MLCMatMul/ReadVariableOp^lstm_2/MLCLSTM/ReadVariableOp ^lstm_2/MLCLSTM/ReadVariableOp_1 ^lstm_2/MLCLSTM/ReadVariableOp_2^lstm_3/MLCLSTM/ReadVariableOp ^lstm_3/MLCLSTM/ReadVariableOp_1 ^lstm_3/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

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
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/MLCMatMul/ReadVariableOp dense_3/MLCMatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/MLCMatMul/ReadVariableOp dense_4/MLCMatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/MLCMatMul/ReadVariableOp dense_5/MLCMatMul/ReadVariableOp2>
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
?
}
(__inference_conv2d_2_layer_call_fn_34260

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_331582
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
?
?
&__inference_lstm_2_layer_call_fn_34474

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
A__inference_lstm_2_layer_call_and_return_conditional_losses_333012
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
?!
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_32913

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
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32744

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
A__inference_lstm_3_layer_call_and_return_conditional_losses_34541
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
? 
?
A__inference_lstm_2_layer_call_and_return_conditional_losses_33335

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
?
&__inference_lstm_2_layer_call_fn_34384
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_328682
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
?
?
F__inference_concatenate_layer_call_and_return_conditional_losses_34278
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
?
K
/__inference_max_pooling2d_2_layer_call_fn_32762

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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_327562
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_33427

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
A__inference_conv2d_layer_call_and_return_conditional_losses_34193

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
B__inference_dense_5_layer_call_and_return_conditional_losses_33504

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02
MLCMatMul/ReadVariableOp?
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

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
?
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32756

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
serving_default_conv2d_input:0??????????<
dense_51
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "name": "conv2d_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}, "name": "conv2d_1_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["conv2d_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d_1_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}, "name": "conv2d_2_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_2_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["max_pooling2d", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["max_pooling2d_1", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["activation_2", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["tf.math.multiply", 0, 0, {}], ["tf.math.multiply_1", 0, 0, {}], ["tf.math.multiply_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze", "inbound_nodes": [["max_pooling2d_2", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["tf.compat.v1.squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}], "input_layers": [["conv2d_input", 0, 0], ["conv2d_1_input", 0, 0], ["conv2d_2_input", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 240, 5]}, {"class_name": "TensorShape", "items": [null, 1, 240, 5]}, {"class_name": "TensorShape", "items": [null, 1, 240, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}, "name": "conv2d_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}, "name": "conv2d_1_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["conv2d_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d_1_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}, "name": "conv2d_2_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_2_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply", "inbound_nodes": [["max_pooling2d", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_1", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_1", "inbound_nodes": [["max_pooling2d_1", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_2", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_2", "inbound_nodes": [["activation_2", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["tf.math.multiply", 0, 0, {}], ["tf.math.multiply_1", 0, 0, {}], ["tf.math.multiply_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze", "inbound_nodes": [["max_pooling2d_2", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_2", "inbound_nodes": [[["tf.compat.v1.squeeze", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["lstm_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_3", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["lstm_3", 0, 0, {}]]]}], "input_layers": [["conv2d_input", 0, 0], ["conv2d_1_input", 0, 0], ["conv2d_2_input", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_1_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}
?


kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 240, 5]}}
?


#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 240, 5]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_2_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 240, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 240, 5]}}
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
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
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 240, 32]}, {"class_name": "TensorShape", "items": [null, 1, 240, 32]}, {"class_name": "TensorShape", "items": [null, 1, 240, 32]}]}
?

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 240, 96]}}
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
T	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.squeeze", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.squeeze", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}}
?
Ucell
V
state_spec
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 48]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 48]}}
?

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 200]}}
?
acell
b
state_spec
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_3", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 30]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 240, 30]}}
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 180, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
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
xlayer_metrics
	variables
trainable_variables
ymetrics
regularization_losses

zlayers
{non_trainable_variables
|layer_regularization_losses
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
}layer_metrics
	variables
 trainable_variables
~metrics
!regularization_losses

layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
%	variables
&trainable_variables
?metrics
'regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
)	variables
*trainable_variables
?metrics
+regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
-	variables
.trainable_variables
?metrics
/regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
3	variables
4trainable_variables
?metrics
5regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
7	variables
8trainable_variables
?metrics
9regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
;	variables
<trainable_variables
?metrics
=regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
?	variables
@trainable_variables
?metrics
Aregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
F	variables
Gtrainable_variables
?metrics
Hregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :`02dense_3/kernel
:02dense_3/bias
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
?layer_metrics
L	variables
Mtrainable_variables
?metrics
Nregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?layer_metrics
P	variables
Qtrainable_variables
?metrics
Rregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
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
?layer_metrics
W	variables
?states
Xtrainable_variables
?metrics
Yregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_4/kernel
:2dense_4/bias
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
?layer_metrics
]	variables
^trainable_variables
?metrics
_regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

ukernel
vrecurrent_kernel
wbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
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
?layer_metrics
c	variables
?states
dtrainable_variables
?metrics
eregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	<?2dense_5/kernel
:?2dense_5/bias
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
?layer_metrics
i	variables
jtrainable_variables
?metrics
kregularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
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
trackable_dict_wrapper
8
?0
?1
?2"
trackable_list_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
?layer_metrics
?	variables
?trainable_variables
?metrics
?regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
U0"
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
?layer_metrics
?	variables
?trainable_variables
?metrics
?regularization_losses
?layers
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
a0"
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
%:#`02Adam/dense_3/kernel/m
:02Adam/dense_3/bias/m
&:$	?2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
&:$	<?2Adam/dense_5/kernel/m
 :?2Adam/dense_5/bias/m
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
%:#`02Adam/dense_3/kernel/v
:02Adam/dense_3/bias/v
&:$	?2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
&:$	<?2Adam/dense_5/kernel/v
 :?2Adam/dense_5/bias/v
1:/	0?2 Adam/lstm_2/lstm_cell_4/kernel/v
<::
??2*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v
+:)?2Adam/lstm_2/lstm_cell_4/bias/v
1:/	?2 Adam/lstm_3/lstm_cell_5/kernel/v
;:9	<?2*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v
+:)?2Adam/lstm_3/lstm_cell_5/bias/v
?2?
%__inference_model_layer_call_fn_34140
%__inference_model_layer_call_fn_33693
%__inference_model_layer_call_fn_34183
%__inference_model_layer_call_fn_33800?
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
 __inference__wrapped_model_32726?
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
?2?
@__inference_model_layer_call_and_return_conditional_losses_33975
@__inference_model_layer_call_and_return_conditional_losses_33521
@__inference_model_layer_call_and_return_conditional_losses_34097
@__inference_model_layer_call_and_return_conditional_losses_33585?
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
?2?
&__inference_conv2d_layer_call_fn_34202?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_34193?
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
(__inference_conv2d_1_layer_call_fn_34221?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_34212?
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
*__inference_activation_layer_call_fn_34231?
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
E__inference_activation_layer_call_and_return_conditional_losses_34226?
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
,__inference_activation_1_layer_call_fn_34241?
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
G__inference_activation_1_layer_call_and_return_conditional_losses_34236?
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
(__inference_conv2d_2_layer_call_fn_34260?
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_34251?
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
-__inference_max_pooling2d_layer_call_fn_32738?
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_32732?
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
/__inference_max_pooling2d_1_layer_call_fn_32750?
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32744?
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
,__inference_activation_2_layer_call_fn_34270?
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
G__inference_activation_2_layer_call_and_return_conditional_losses_34265?
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
+__inference_concatenate_layer_call_fn_34285?
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
F__inference_concatenate_layer_call_and_return_conditional_losses_34278?
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
'__inference_dense_3_layer_call_fn_34305?
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
B__inference_dense_3_layer_call_and_return_conditional_losses_34296?
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
/__inference_max_pooling2d_2_layer_call_fn_32762?
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32756?
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
&__inference_lstm_2_layer_call_fn_34384
&__inference_lstm_2_layer_call_fn_34485
&__inference_lstm_2_layer_call_fn_34474
&__inference_lstm_2_layer_call_fn_34395?
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
?2?
A__inference_lstm_2_layer_call_and_return_conditional_losses_34339
A__inference_lstm_2_layer_call_and_return_conditional_losses_34463
A__inference_lstm_2_layer_call_and_return_conditional_losses_34373
A__inference_lstm_2_layer_call_and_return_conditional_losses_34429?
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
'__inference_dense_4_layer_call_fn_34505?
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
B__inference_dense_4_layer_call_and_return_conditional_losses_34496?
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
&__inference_lstm_3_layer_call_fn_34588
&__inference_lstm_3_layer_call_fn_34693
&__inference_lstm_3_layer_call_fn_34682
&__inference_lstm_3_layer_call_fn_34599?
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
?2?
A__inference_lstm_3_layer_call_and_return_conditional_losses_34635
A__inference_lstm_3_layer_call_and_return_conditional_losses_34577
A__inference_lstm_3_layer_call_and_return_conditional_losses_34541
A__inference_lstm_3_layer_call_and_return_conditional_losses_34671?
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
'__inference_dense_5_layer_call_fn_34713?
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
B__inference_dense_5_layer_call_and_return_conditional_losses_34704?
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
#__inference_signature_wrapper_33853conv2d_1_inputconv2d_2_inputconv2d_input"?
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
 __inference__wrapped_model_32726?#$12JKrst[\uvwgh???
???
???
/?,
conv2d_input??????????
1?.
conv2d_1_input??????????
1?.
conv2d_2_input??????????
? "2?/
-
dense_5"?
dense_5???????????
G__inference_activation_1_layer_call_and_return_conditional_losses_34236j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_1_layer_call_fn_34241]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
G__inference_activation_2_layer_call_and_return_conditional_losses_34265j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_2_layer_call_fn_34270]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
E__inference_activation_layer_call_and_return_conditional_losses_34226j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
*__inference_activation_layer_call_fn_34231]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
F__inference_concatenate_layer_call_and_return_conditional_losses_34278????
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
+__inference_concatenate_layer_call_fn_34285????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? "!???????????`?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_34212n#$8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_1_layer_call_fn_34221a#$8?5
.?+
)?&
inputs??????????
? "!??????????? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_34251n128?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_2_layer_call_fn_34260a128?5
.?+
)?&
inputs??????????
? "!??????????? ?
A__inference_conv2d_layer_call_and_return_conditional_losses_34193n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
&__inference_conv2d_layer_call_fn_34202a8?5
.?+
)?&
inputs??????????
? "!??????????? ?
B__inference_dense_3_layer_call_and_return_conditional_losses_34296nJK8?5
.?+
)?&
inputs??????????`
? ".?+
$?!
0??????????0
? ?
'__inference_dense_3_layer_call_fn_34305aJK8?5
.?+
)?&
inputs??????????`
? "!???????????0?
B__inference_dense_4_layer_call_and_return_conditional_losses_34496g[\5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????
? ?
'__inference_dense_4_layer_call_fn_34505Z[\5?2
+?(
&?#
inputs???????????
? "????????????
B__inference_dense_5_layer_call_and_return_conditional_losses_34704]gh/?,
%?"
 ?
inputs?????????<
? "&?#
?
0??????????
? {
'__inference_dense_5_layer_call_fn_34713Pgh/?,
%?"
 ?
inputs?????????<
? "????????????
A__inference_lstm_2_layer_call_and_return_conditional_losses_34339?rstO?L
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_34373?rstO?L
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_34429trst@?=
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
A__inference_lstm_2_layer_call_and_return_conditional_losses_34463trst@?=
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
&__inference_lstm_2_layer_call_fn_34384~rstO?L
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
&__inference_lstm_2_layer_call_fn_34395~rstO?L
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
&__inference_lstm_2_layer_call_fn_34474grst@?=
6?3
%?"
inputs??????????0

 
p

 
? "?????????????
&__inference_lstm_2_layer_call_fn_34485grst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "?????????????
A__inference_lstm_3_layer_call_and_return_conditional_losses_34541}uvwO?L
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_34577}uvwO?L
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_34635nuvw@?=
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
A__inference_lstm_3_layer_call_and_return_conditional_losses_34671nuvw@?=
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
&__inference_lstm_3_layer_call_fn_34588puvwO?L
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
&__inference_lstm_3_layer_call_fn_34599puvwO?L
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
&__inference_lstm_3_layer_call_fn_34682auvw@?=
6?3
%?"
inputs??????????

 
p

 
? "??????????<?
&__inference_lstm_3_layer_call_fn_34693auvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "??????????<?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_32744?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_32750?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_32756?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_2_layer_call_fn_32762?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_32732?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_32738?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
@__inference_model_layer_call_and_return_conditional_losses_33521?#$12JKrst[\uvwgh???
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
? "&?#
?
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_33585?#$12JKrst[\uvwgh???
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
? "&?#
?
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_33975?#$12JKrst[\uvwgh???
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
? "&?#
?
0??????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_34097?#$12JKrst[\uvwgh???
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
? "&?#
?
0??????????
? ?
%__inference_model_layer_call_fn_33693?#$12JKrst[\uvwgh???
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
? "????????????
%__inference_model_layer_call_fn_33800?#$12JKrst[\uvwgh???
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
? "????????????
%__inference_model_layer_call_fn_34140?#$12JKrst[\uvwgh???
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
? "????????????
%__inference_model_layer_call_fn_34183?#$12JKrst[\uvwgh???
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
? "????????????
#__inference_signature_wrapper_33853?#$12JKrst[\uvwgh???
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
conv2d_input??????????"2?/
-
dense_5"?
dense_5??????????