??
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
?
conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
: *
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
: *
dtype0
?
conv2d_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_16/kernel
}
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*&
_output_shapes
: *
dtype0
t
conv2d_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_16/bias
m
"conv2d_16/bias/Read/ReadVariableOpReadVariableOpconv2d_16/bias*
_output_shapes
: *
dtype0
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_17/kernel
}
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*&
_output_shapes
: *
dtype0
t
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_17/bias
m
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes
: *
dtype0
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0* 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

:`0*
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
:0*
dtype0
{
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_40/kernel
t
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes
:	?*
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
:*
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H* 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

:<H*
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
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
lstm_22/lstm_cell_44/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*,
shared_namelstm_22/lstm_cell_44/kernel
?
/lstm_22/lstm_cell_44/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_44/kernel*
_output_shapes
:	0?*
dtype0
?
%lstm_22/lstm_cell_44/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_22/lstm_cell_44/recurrent_kernel
?
9lstm_22/lstm_cell_44/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_44/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_22/lstm_cell_44/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_22/lstm_cell_44/bias
?
-lstm_22/lstm_cell_44/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_44/bias*
_output_shapes	
:?*
dtype0
?
lstm_23/lstm_cell_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namelstm_23/lstm_cell_45/kernel
?
/lstm_23/lstm_cell_45/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_45/kernel*
_output_shapes
:	?*
dtype0
?
%lstm_23/lstm_cell_45/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*6
shared_name'%lstm_23/lstm_cell_45/recurrent_kernel
?
9lstm_23/lstm_cell_45/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_23/lstm_cell_45/recurrent_kernel*
_output_shapes
:	<?*
dtype0
?
lstm_23/lstm_cell_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_23/lstm_cell_45/bias
?
-lstm_23/lstm_cell_45/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_45/bias*
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
Adam/conv2d_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_15/kernel/m
?
+Adam/conv2d_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_15/bias/m
{
)Adam/conv2d_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/m
?
+Adam/conv2d_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/m
{
)Adam/conv2d_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_17/kernel/m
?
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_17/bias/m
{
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*'
shared_nameAdam/dense_39/kernel/m
?
*Adam/dense_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/m*
_output_shapes

:`0*
dtype0
?
Adam/dense_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_39/bias/m
y
(Adam/dense_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_40/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_40/kernel/m
?
*Adam/dense_40/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_40/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_40/bias/m
y
(Adam/dense_40/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H*'
shared_nameAdam/dense_41/kernel/m
?
*Adam/dense_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/m*
_output_shapes

:<H*
dtype0
?
Adam/dense_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_41/bias/m
y
(Adam/dense_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/m*
_output_shapes
:H*
dtype0
?
"Adam/lstm_22/lstm_cell_44/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*3
shared_name$"Adam/lstm_22/lstm_cell_44/kernel/m
?
6Adam/lstm_22/lstm_cell_44/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_44/kernel/m*
_output_shapes
:	0?*
dtype0
?
,Adam/lstm_22/lstm_cell_44/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_22/lstm_cell_44/recurrent_kernel/m
?
@Adam/lstm_22/lstm_cell_44/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_44/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_22/lstm_cell_44/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_22/lstm_cell_44/bias/m
?
4Adam/lstm_22/lstm_cell_44/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_44/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_23/lstm_cell_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_23/lstm_cell_45/kernel/m
?
6Adam/lstm_23/lstm_cell_45/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_23/lstm_cell_45/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_23/lstm_cell_45/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*=
shared_name.,Adam/lstm_23/lstm_cell_45/recurrent_kernel/m
?
@Adam/lstm_23/lstm_cell_45/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_23/lstm_cell_45/recurrent_kernel/m*
_output_shapes
:	<?*
dtype0
?
 Adam/lstm_23/lstm_cell_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_23/lstm_cell_45/bias/m
?
4Adam/lstm_23/lstm_cell_45/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_23/lstm_cell_45/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_15/kernel/v
?
+Adam/conv2d_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_15/bias/v
{
)Adam/conv2d_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_15/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_16/kernel/v
?
+Adam/conv2d_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_16/bias/v
{
)Adam/conv2d_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_16/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_17/kernel/v
?
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_17/bias/v
{
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*'
shared_nameAdam/dense_39/kernel/v
?
*Adam/dense_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/kernel/v*
_output_shapes

:`0*
dtype0
?
Adam/dense_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/dense_39/bias/v
y
(Adam/dense_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_39/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_40/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_40/kernel/v
?
*Adam/dense_40/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_40/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_40/bias/v
y
(Adam/dense_40/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_40/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<H*'
shared_nameAdam/dense_41/kernel/v
?
*Adam/dense_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/kernel/v*
_output_shapes

:<H*
dtype0
?
Adam/dense_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*%
shared_nameAdam/dense_41/bias/v
y
(Adam/dense_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_41/bias/v*
_output_shapes
:H*
dtype0
?
"Adam/lstm_22/lstm_cell_44/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*3
shared_name$"Adam/lstm_22/lstm_cell_44/kernel/v
?
6Adam/lstm_22/lstm_cell_44/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_44/kernel/v*
_output_shapes
:	0?*
dtype0
?
,Adam/lstm_22/lstm_cell_44/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_22/lstm_cell_44/recurrent_kernel/v
?
@Adam/lstm_22/lstm_cell_44/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_44/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_22/lstm_cell_44/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_22/lstm_cell_44/bias/v
?
4Adam/lstm_22/lstm_cell_44/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_44/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_23/lstm_cell_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/lstm_23/lstm_cell_45/kernel/v
?
6Adam/lstm_23/lstm_cell_45/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_23/lstm_cell_45/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/lstm_23/lstm_cell_45/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*=
shared_name.,Adam/lstm_23/lstm_cell_45/recurrent_kernel/v
?
@Adam/lstm_23/lstm_cell_45/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_23/lstm_cell_45/recurrent_kernel/v*
_output_shapes
:	<?*
dtype0
?
 Adam/lstm_23/lstm_cell_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_23/lstm_cell_45/bias/v
?
4Adam/lstm_23/lstm_cell_45/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_23/lstm_cell_45/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?s
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?r
value?rB?r B?r
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
 
R
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
R
;regularization_losses
<	variables
=trainable_variables
>	keras_api
R
?regularization_losses
@	variables
Atrainable_variables
B	keras_api

C	keras_api

D	keras_api

E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api

T	keras_api
l
Ucell
V
state_spec
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
l
acell
b
state_spec
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?1m?2m?Jm?Km?[m?\m?gm?hm?rm?sm?tm?um?vm?wm?v?v?#v?$v?1v?2v?Jv?Kv?[v?\v?gv?hv?rv?sv?tv?uv?vv?wv?
 
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
?
regularization_losses
xlayer_regularization_losses
ylayer_metrics
	variables
znon_trainable_variables
trainable_variables
{metrics

|layers
 
\Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
}layer_regularization_losses
~layer_metrics
 	variables
non_trainable_variables
!trainable_variables
?metrics
?layers
\Z
VARIABLE_VALUEconv2d_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
%regularization_losses
 ?layer_regularization_losses
?layer_metrics
&	variables
?non_trainable_variables
'trainable_variables
?metrics
?layers
 
 
 
?
)regularization_losses
 ?layer_regularization_losses
?layer_metrics
*	variables
?non_trainable_variables
+trainable_variables
?metrics
?layers
 
 
 
?
-regularization_losses
 ?layer_regularization_losses
?layer_metrics
.	variables
?non_trainable_variables
/trainable_variables
?metrics
?layers
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
3regularization_losses
 ?layer_regularization_losses
?layer_metrics
4	variables
?non_trainable_variables
5trainable_variables
?metrics
?layers
 
 
 
?
7regularization_losses
 ?layer_regularization_losses
?layer_metrics
8	variables
?non_trainable_variables
9trainable_variables
?metrics
?layers
 
 
 
?
;regularization_losses
 ?layer_regularization_losses
?layer_metrics
<	variables
?non_trainable_variables
=trainable_variables
?metrics
?layers
 
 
 
?
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
@	variables
?non_trainable_variables
Atrainable_variables
?metrics
?layers
 
 
 
 
 
 
?
Fregularization_losses
 ?layer_regularization_losses
?layer_metrics
G	variables
?non_trainable_variables
Htrainable_variables
?metrics
?layers
[Y
VARIABLE_VALUEdense_39/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
?
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
M	variables
?non_trainable_variables
Ntrainable_variables
?metrics
?layers
 
 
 
?
Pregularization_losses
 ?layer_regularization_losses
?layer_metrics
Q	variables
?non_trainable_variables
Rtrainable_variables
?metrics
?layers
 
?

rkernel
srecurrent_kernel
tbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
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
?
Wregularization_losses
 ?layer_regularization_losses
?layer_metrics
X	variables
?non_trainable_variables
Ytrainable_variables
?metrics
?layers
?states
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
?
]regularization_losses
 ?layer_regularization_losses
?layer_metrics
^	variables
?non_trainable_variables
_trainable_variables
?metrics
?layers
?

ukernel
vrecurrent_kernel
wbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
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
?
cregularization_losses
 ?layer_regularization_losses
?layer_metrics
d	variables
?non_trainable_variables
etrainable_variables
?metrics
?layers
?states
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
?
iregularization_losses
 ?layer_regularization_losses
?layer_metrics
j	variables
?non_trainable_variables
ktrainable_variables
?metrics
?layers
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
WU
VARIABLE_VALUElstm_22/lstm_cell_44/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_22/lstm_cell_44/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_22/lstm_cell_44/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUElstm_23/lstm_cell_45/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%lstm_23/lstm_cell_45/recurrent_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElstm_23/lstm_cell_45/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
 
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

r0
s1
t2

r0
s1
t2
?
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?layers
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
?
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?layers
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
}
VARIABLE_VALUEAdam/conv2d_15/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_15/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_16/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_16/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_44/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_44/recurrent_kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_22/lstm_cell_44/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/lstm_23/lstm_cell_45/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_23/lstm_cell_45/recurrent_kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_23/lstm_cell_45/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_15/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_15/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_16/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_16/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_39/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_39/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_40/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_40/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_41/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_41/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_44/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_44/recurrent_kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_22/lstm_cell_44/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/lstm_23/lstm_cell_45/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_23/lstm_cell_45/recurrent_kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/lstm_23/lstm_cell_45/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_15_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_16_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_17_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_15_inputserving_default_conv2d_16_inputserving_default_conv2d_17_inputconv2d_16/kernelconv2d_16/biasconv2d_15/kernelconv2d_15/biasconv2d_17/kernelconv2d_17/biasdense_39/kerneldense_39/biaslstm_22/lstm_cell_44/kernel%lstm_22/lstm_cell_44/recurrent_kernellstm_22/lstm_cell_44/biasdense_40/kerneldense_40/biaslstm_23/lstm_cell_45/kernel%lstm_23/lstm_cell_45/recurrent_kernellstm_23/lstm_cell_45/biasdense_41/kerneldense_41/bias* 
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_279095
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp"conv2d_16/bias/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_22/lstm_cell_44/kernel/Read/ReadVariableOp9lstm_22/lstm_cell_44/recurrent_kernel/Read/ReadVariableOp-lstm_22/lstm_cell_44/bias/Read/ReadVariableOp/lstm_23/lstm_cell_45/kernel/Read/ReadVariableOp9lstm_23/lstm_cell_45/recurrent_kernel/Read/ReadVariableOp-lstm_23/lstm_cell_45/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/conv2d_15/kernel/m/Read/ReadVariableOp)Adam/conv2d_15/bias/m/Read/ReadVariableOp+Adam/conv2d_16/kernel/m/Read/ReadVariableOp)Adam/conv2d_16/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp*Adam/dense_39/kernel/m/Read/ReadVariableOp(Adam/dense_39/bias/m/Read/ReadVariableOp*Adam/dense_40/kernel/m/Read/ReadVariableOp(Adam/dense_40/bias/m/Read/ReadVariableOp*Adam/dense_41/kernel/m/Read/ReadVariableOp(Adam/dense_41/bias/m/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_44/kernel/m/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_44/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_44/bias/m/Read/ReadVariableOp6Adam/lstm_23/lstm_cell_45/kernel/m/Read/ReadVariableOp@Adam/lstm_23/lstm_cell_45/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_23/lstm_cell_45/bias/m/Read/ReadVariableOp+Adam/conv2d_15/kernel/v/Read/ReadVariableOp)Adam/conv2d_15/bias/v/Read/ReadVariableOp+Adam/conv2d_16/kernel/v/Read/ReadVariableOp)Adam/conv2d_16/bias/v/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp*Adam/dense_39/kernel/v/Read/ReadVariableOp(Adam/dense_39/bias/v/Read/ReadVariableOp*Adam/dense_40/kernel/v/Read/ReadVariableOp(Adam/dense_40/bias/v/Read/ReadVariableOp*Adam/dense_41/kernel/v/Read/ReadVariableOp(Adam/dense_41/bias/v/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_44/kernel/v/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_44/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_44/bias/v/Read/ReadVariableOp6Adam/lstm_23/lstm_cell_45/kernel/v/Read/ReadVariableOp@Adam/lstm_23/lstm_cell_45/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_23/lstm_cell_45/bias/v/Read/ReadVariableOpConst*N
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
GPU 2J 8? *(
f#R!
__inference__traced_save_280175
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_15/kernelconv2d_15/biasconv2d_16/kernelconv2d_16/biasconv2d_17/kernelconv2d_17/biasdense_39/kerneldense_39/biasdense_40/kerneldense_40/biasdense_41/kerneldense_41/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_22/lstm_cell_44/kernel%lstm_22/lstm_cell_44/recurrent_kernellstm_22/lstm_cell_44/biaslstm_23/lstm_cell_45/kernel%lstm_23/lstm_cell_45/recurrent_kernellstm_23/lstm_cell_45/biastotalcounttotal_1count_1total_2count_2Adam/conv2d_15/kernel/mAdam/conv2d_15/bias/mAdam/conv2d_16/kernel/mAdam/conv2d_16/bias/mAdam/conv2d_17/kernel/mAdam/conv2d_17/bias/mAdam/dense_39/kernel/mAdam/dense_39/bias/mAdam/dense_40/kernel/mAdam/dense_40/bias/mAdam/dense_41/kernel/mAdam/dense_41/bias/m"Adam/lstm_22/lstm_cell_44/kernel/m,Adam/lstm_22/lstm_cell_44/recurrent_kernel/m Adam/lstm_22/lstm_cell_44/bias/m"Adam/lstm_23/lstm_cell_45/kernel/m,Adam/lstm_23/lstm_cell_45/recurrent_kernel/m Adam/lstm_23/lstm_cell_45/bias/mAdam/conv2d_15/kernel/vAdam/conv2d_15/bias/vAdam/conv2d_16/kernel/vAdam/conv2d_16/bias/vAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/vAdam/dense_39/kernel/vAdam/dense_39/bias/vAdam/dense_40/kernel/vAdam/dense_40/bias/vAdam/dense_41/kernel/vAdam/dense_41/bias/v"Adam/lstm_22/lstm_cell_44/kernel/v,Adam/lstm_22/lstm_cell_44/recurrent_kernel/v Adam/lstm_22/lstm_cell_44/bias/v"Adam/lstm_23/lstm_cell_45/kernel/v,Adam/lstm_23/lstm_cell_45/recurrent_kernel/v Adam/lstm_23/lstm_cell_45/bias/v*M
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_280380??
?
~
)__inference_dense_40_layer_call_fn_279747

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
GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2786182
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
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_279217
inputs_0
inputs_1
inputs_2,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource.
*dense_39_mlcmatmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'lstm_22_mlclstm_readvariableop_resource-
)lstm_22_mlclstm_readvariableop_1_resource-
)lstm_22_mlclstm_readvariableop_2_resource.
*dense_40_mlcmatmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'lstm_23_mlclstm_readvariableop_resource-
)lstm_23_mlclstm_readvariableop_1_resource-
)lstm_23_mlclstm_readvariableop_2_resource.
*dense_41_mlcmatmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?!dense_39/MLCMatMul/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?!dense_40/MLCMatMul/ReadVariableOp?dense_41/BiasAdd/ReadVariableOp?!dense_41/MLCMatMul/ReadVariableOp?lstm_22/MLCLSTM/ReadVariableOp? lstm_22/MLCLSTM/ReadVariableOp_1? lstm_22/MLCLSTM/ReadVariableOp_2?lstm_23/MLCLSTM/ReadVariableOp? lstm_23/MLCLSTM/ReadVariableOp_1? lstm_23/MLCLSTM/ReadVariableOp_2?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D	MLCConv2Dinputs_1'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_16/BiasAdd?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D	MLCConv2Dinputs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_15/BiasAdd?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D	MLCConv2Dinputs_2'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_17/BiasAdd?
activation_28/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_28/Relu?
activation_27/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_27/Relu?
activation_29/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_29/Relu?
max_pooling2d_16/MaxPoolMaxPool activation_28/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool?
max_pooling2d_15/MaxPoolMaxPool activation_27/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul!max_pooling2d_15/MaxPool:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul!max_pooling2d_16/MaxPool:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul activation_29/Relu:activations:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mulx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2tf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0"concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate_5/concat?
!dense_39/MLCMatMul/ReadVariableOpReadVariableOp*dense_39_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02#
!dense_39/MLCMatMul/ReadVariableOp?
dense_39/MLCMatMul	MLCMatMulconcatenate_5/concat:output:0)dense_39/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_39/MLCMatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MLCMatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_39/BiasAdd|
dense_39/TanhTanhdense_39/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_39/Tanh?
max_pooling2d_17/MaxPoolMaxPooldense_39/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool?
tf.compat.v1.squeeze_5/SqueezeSqueeze!max_pooling2d_17/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeezeu
lstm_22/ShapeShape'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_22/Shape?
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack?
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1?
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2?
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slicem
lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/mul/y?
lstm_22/zeros/mulMullstm_22/strided_slice:output:0lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/mulo
lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/Less/y?
lstm_22/zeros/LessLesslstm_22/zeros/mul:z:0lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/Lesss
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/packed/1?
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const?
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/mul/y?
lstm_22/zeros_1/mulMullstm_22/strided_slice:output:0lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/muls
lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/Less/y?
lstm_22/zeros_1/LessLesslstm_22/zeros_1/mul:z:0lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/Lessw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/packed/1?
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const?
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_22/zeros_1}
lstm_22/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/MLCLSTM/hidden_size}
lstm_22/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/MLCLSTM/output_size?
lstm_22/MLCLSTM/ReadVariableOpReadVariableOp'lstm_22_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02 
lstm_22/MLCLSTM/ReadVariableOp?
 lstm_22/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_22_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02"
 lstm_22/MLCLSTM/ReadVariableOp_1?
 lstm_22/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_22_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_22/MLCLSTM/ReadVariableOp_2?
lstm_22/MLCLSTMMLCLSTM'tf.compat.v1.squeeze_5/Squeeze:output:0$lstm_22/MLCLSTM/hidden_size:output:0$lstm_22/MLCLSTM/output_size:output:0&lstm_22/MLCLSTM/ReadVariableOp:value:0(lstm_22/MLCLSTM/ReadVariableOp_1:value:0(lstm_22/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_22/MLCLSTM?
!dense_40/MLCMatMul/ReadVariableOpReadVariableOp*dense_40_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_40/MLCMatMul/ReadVariableOp?
dense_40/MLCMatMul	MLCMatMullstm_22/MLCLSTM:output:0)dense_40/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_40/MLCMatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MLCMatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_40/BiasAddx
dense_40/TanhTanhdense_40/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_40/Tanh_
lstm_23/ShapeShapedense_40/Tanh:y:0*
T0*
_output_shapes
:2
lstm_23/Shape?
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stack?
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1?
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2?
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicel
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros/mul/y?
lstm_23/zeros/mulMullstm_23/strided_slice:output:0lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/mulo
lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_23/zeros/Less/y?
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lessr
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros/packed/1?
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros/packedo
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros/Const?
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/zerosp
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros_1/mul/y?
lstm_23/zeros_1/mulMullstm_23/strided_slice:output:0lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/muls
lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_23/zeros_1/Less/y?
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessv
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros_1/packed/1?
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros_1/packeds
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros_1/Const?
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/zeros_1|
lstm_23/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/MLCLSTM/hidden_size|
lstm_23/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/MLCLSTM/output_size?
lstm_23/MLCLSTM/ReadVariableOpReadVariableOp'lstm_23_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_23/MLCLSTM/ReadVariableOp?
 lstm_23/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_23_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02"
 lstm_23/MLCLSTM/ReadVariableOp_1?
 lstm_23/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_23_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_23/MLCLSTM/ReadVariableOp_2?
lstm_23/MLCLSTMMLCLSTMdense_40/Tanh:y:0$lstm_23/MLCLSTM/hidden_size:output:0$lstm_23/MLCLSTM/output_size:output:0&lstm_23/MLCLSTM/ReadVariableOp:value:0(lstm_23/MLCLSTM/ReadVariableOp_1:value:0(lstm_23/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_23/MLCLSTM
lstm_23/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_23/Reshape/shape?
lstm_23/ReshapeReshapelstm_23/MLCLSTM:output:0lstm_23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/Reshape?
!dense_41/MLCMatMul/ReadVariableOpReadVariableOp*dense_41_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02#
!dense_41/MLCMatMul/ReadVariableOp?
dense_41/MLCMatMul	MLCMatMullstm_23/Reshape:output:0)dense_41/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_41/MLCMatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MLCMatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_41/BiasAdds
dense_41/TanhTanhdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_41/Tanh?
IdentityIdentitydense_41/Tanh:y:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp"^dense_39/MLCMatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp"^dense_40/MLCMatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp"^dense_41/MLCMatMul/ReadVariableOp^lstm_22/MLCLSTM/ReadVariableOp!^lstm_22/MLCLSTM/ReadVariableOp_1!^lstm_22/MLCLSTM/ReadVariableOp_2^lstm_23/MLCLSTM/ReadVariableOp!^lstm_23/MLCLSTM/ReadVariableOp_1!^lstm_23/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2F
!dense_39/MLCMatMul/ReadVariableOp!dense_39/MLCMatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2F
!dense_40/MLCMatMul/ReadVariableOp!dense_40/MLCMatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2F
!dense_41/MLCMatMul/ReadVariableOp!dense_41/MLCMatMul/ReadVariableOp2@
lstm_22/MLCLSTM/ReadVariableOplstm_22/MLCLSTM/ReadVariableOp2D
 lstm_22/MLCLSTM/ReadVariableOp_1 lstm_22/MLCLSTM/ReadVariableOp_12D
 lstm_22/MLCLSTM/ReadVariableOp_2 lstm_22/MLCLSTM/ReadVariableOp_22@
lstm_23/MLCLSTM/ReadVariableOplstm_23/MLCLSTM/ReadVariableOp2D
 lstm_23/MLCLSTM/ReadVariableOp_1 lstm_23/MLCLSTM/ReadVariableOp_12D
 lstm_23/MLCLSTM/ReadVariableOp_2 lstm_23/MLCLSTM/ReadVariableOp_2:Z V
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
?"
?
C__inference_lstm_23_layer_call_and_return_conditional_losses_278669

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
C__inference_lstm_22_layer_call_and_return_conditional_losses_279581
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
?L
?
C__inference_model_5_layer_call_and_return_conditional_losses_278896

inputs
inputs_1
inputs_2
conv2d_16_278837
conv2d_16_278839
conv2d_15_278842
conv2d_15_278844
conv2d_17_278847
conv2d_17_278849
dense_39_278864
dense_39_278866
lstm_22_278871
lstm_22_278873
lstm_22_278875
dense_40_278878
dense_40_278880
lstm_23_278883
lstm_23_278885
lstm_23_278887
dense_41_278890
dense_41_278892
identity??!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?lstm_22/StatefulPartitionedCall?lstm_23/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_16_278837conv2d_16_278839*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2783482#
!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_278842conv2d_15_278844*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2783742#
!conv2d_15/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_17_278847conv2d_17_278849*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2784002#
!conv2d_17/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_2784212
activation_28/PartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_27_layer_call_and_return_conditional_losses_2784342
activation_27/PartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_2784472
activation_29/PartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_2779862"
 max_pooling2d_16/PartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2779742"
 max_pooling2d_15/PartitionedCall{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul)max_pooling2d_15/PartitionedCall:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul)max_pooling2d_16/PartitionedCall:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul&activation_29/PartitionedCall:output:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mul?
concatenate_5/PartitionedCallPartitionedCalltf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0*
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
GPU 2J 8? *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_2784712
concatenate_5/PartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_278864dense_39_278866*
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
GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2784922"
 dense_39/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_2779982"
 max_pooling2d_17/PartitionedCall?
tf.compat.v1.squeeze_5/SqueezeSqueeze)max_pooling2d_17/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeeze?
lstm_22/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_5/Squeeze:output:0lstm_22_278871lstm_22_278873lstm_22_278875*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785432!
lstm_22/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_40_278878dense_40_278880*
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
GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2786182"
 dense_40/StatefulPartitionedCall?
lstm_23/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_23_278883lstm_23_278885lstm_23_278887*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2786692!
lstm_23/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0dense_41_278890dense_41_278892*
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
GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_2787462"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:X T
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_278543

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
?M
?
C__inference_model_5_layer_call_and_return_conditional_losses_278763
conv2d_15_input
conv2d_16_input
conv2d_17_input
conv2d_16_278359
conv2d_16_278361
conv2d_15_278385
conv2d_15_278387
conv2d_17_278411
conv2d_17_278413
dense_39_278503
dense_39_278505
lstm_22_278600
lstm_22_278602
lstm_22_278604
dense_40_278629
dense_40_278631
lstm_23_278728
lstm_23_278730
lstm_23_278732
dense_41_278757
dense_41_278759
identity??!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?lstm_22/StatefulPartitionedCall?lstm_23/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_278359conv2d_16_278361*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2783482#
!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_15_278385conv2d_15_278387*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2783742#
!conv2d_15/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_278411conv2d_17_278413*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2784002#
!conv2d_17/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_2784212
activation_28/PartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_27_layer_call_and_return_conditional_losses_2784342
activation_27/PartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_2784472
activation_29/PartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_2779862"
 max_pooling2d_16/PartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2779742"
 max_pooling2d_15/PartitionedCall{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul)max_pooling2d_15/PartitionedCall:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul)max_pooling2d_16/PartitionedCall:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul&activation_29/PartitionedCall:output:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mul?
concatenate_5/PartitionedCallPartitionedCalltf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0*
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
GPU 2J 8? *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_2784712
concatenate_5/PartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_278503dense_39_278505*
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
GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2784922"
 dense_39/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_2779982"
 max_pooling2d_17/PartitionedCall?
tf.compat.v1.squeeze_5/SqueezeSqueeze)max_pooling2d_17/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeeze?
lstm_22/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_5/Squeeze:output:0lstm_22_278600lstm_22_278602lstm_22_278604*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785432!
lstm_22/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_40_278629dense_40_278631*
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
GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2786182"
 dense_40/StatefulPartitionedCall?
lstm_23/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_23_278728lstm_23_278730lstm_23_278732*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2786692!
lstm_23/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0dense_41_278757dense_41_278759*
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
GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_2787462"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?
?
(__inference_lstm_22_layer_call_fn_279727

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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785772
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
?
?
(__inference_lstm_22_layer_call_fn_279637
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2781552
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
e
I__inference_activation_27_layer_call_and_return_conditional_losses_279468

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
?
(__inference_lstm_22_layer_call_fn_279716

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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785432
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_278110

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
?!
?
C__inference_lstm_22_layer_call_and_return_conditional_losses_279615
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
(__inference_lstm_23_layer_call_fn_279830

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
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2786692
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

?
D__inference_dense_40_layer_call_and_return_conditional_losses_278618

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
?"
?
C__inference_lstm_23_layer_call_and_return_conditional_losses_279913
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
?
J
.__inference_activation_28_layer_call_fn_279483

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
GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_2784212
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_279877
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

?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_278374

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
??
?
!__inference__wrapped_model_277968
conv2d_15_input
conv2d_16_input
conv2d_17_input4
0model_5_conv2d_16_conv2d_readvariableop_resource5
1model_5_conv2d_16_biasadd_readvariableop_resource4
0model_5_conv2d_15_conv2d_readvariableop_resource5
1model_5_conv2d_15_biasadd_readvariableop_resource4
0model_5_conv2d_17_conv2d_readvariableop_resource5
1model_5_conv2d_17_biasadd_readvariableop_resource6
2model_5_dense_39_mlcmatmul_readvariableop_resource4
0model_5_dense_39_biasadd_readvariableop_resource3
/model_5_lstm_22_mlclstm_readvariableop_resource5
1model_5_lstm_22_mlclstm_readvariableop_1_resource5
1model_5_lstm_22_mlclstm_readvariableop_2_resource6
2model_5_dense_40_mlcmatmul_readvariableop_resource4
0model_5_dense_40_biasadd_readvariableop_resource3
/model_5_lstm_23_mlclstm_readvariableop_resource5
1model_5_lstm_23_mlclstm_readvariableop_1_resource5
1model_5_lstm_23_mlclstm_readvariableop_2_resource6
2model_5_dense_41_mlcmatmul_readvariableop_resource4
0model_5_dense_41_biasadd_readvariableop_resource
identity??(model_5/conv2d_15/BiasAdd/ReadVariableOp?'model_5/conv2d_15/Conv2D/ReadVariableOp?(model_5/conv2d_16/BiasAdd/ReadVariableOp?'model_5/conv2d_16/Conv2D/ReadVariableOp?(model_5/conv2d_17/BiasAdd/ReadVariableOp?'model_5/conv2d_17/Conv2D/ReadVariableOp?'model_5/dense_39/BiasAdd/ReadVariableOp?)model_5/dense_39/MLCMatMul/ReadVariableOp?'model_5/dense_40/BiasAdd/ReadVariableOp?)model_5/dense_40/MLCMatMul/ReadVariableOp?'model_5/dense_41/BiasAdd/ReadVariableOp?)model_5/dense_41/MLCMatMul/ReadVariableOp?&model_5/lstm_22/MLCLSTM/ReadVariableOp?(model_5/lstm_22/MLCLSTM/ReadVariableOp_1?(model_5/lstm_22/MLCLSTM/ReadVariableOp_2?&model_5/lstm_23/MLCLSTM/ReadVariableOp?(model_5/lstm_23/MLCLSTM/ReadVariableOp_1?(model_5/lstm_23/MLCLSTM/ReadVariableOp_2?
'model_5/conv2d_16/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_5/conv2d_16/Conv2D/ReadVariableOp?
model_5/conv2d_16/Conv2D	MLCConv2Dconv2d_16_input/model_5/conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_5/conv2d_16/Conv2D?
(model_5/conv2d_16/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv2d_16/BiasAdd/ReadVariableOp?
model_5/conv2d_16/BiasAddBiasAdd!model_5/conv2d_16/Conv2D:output:00model_5/conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_5/conv2d_16/BiasAdd?
'model_5/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_5/conv2d_15/Conv2D/ReadVariableOp?
model_5/conv2d_15/Conv2D	MLCConv2Dconv2d_15_input/model_5/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_5/conv2d_15/Conv2D?
(model_5/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv2d_15/BiasAdd/ReadVariableOp?
model_5/conv2d_15/BiasAddBiasAdd!model_5/conv2d_15/Conv2D:output:00model_5/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_5/conv2d_15/BiasAdd?
'model_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp0model_5_conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_5/conv2d_17/Conv2D/ReadVariableOp?
model_5/conv2d_17/Conv2D	MLCConv2Dconv2d_17_input/model_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_5/conv2d_17/Conv2D?
(model_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp1model_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_5/conv2d_17/BiasAdd/ReadVariableOp?
model_5/conv2d_17/BiasAddBiasAdd!model_5/conv2d_17/Conv2D:output:00model_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_5/conv2d_17/BiasAdd?
model_5/activation_28/ReluRelu"model_5/conv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_5/activation_28/Relu?
model_5/activation_27/ReluRelu"model_5/conv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_5/activation_27/Relu?
model_5/activation_29/ReluRelu"model_5/conv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_5/activation_29/Relu?
 model_5/max_pooling2d_16/MaxPoolMaxPool(model_5/activation_28/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling2d_16/MaxPool?
 model_5/max_pooling2d_15/MaxPoolMaxPool(model_5/activation_27/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling2d_15/MaxPool?
!model_5/tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!model_5/tf.math.multiply_15/Mul/y?
model_5/tf.math.multiply_15/MulMul)model_5/max_pooling2d_15/MaxPool:output:0*model_5/tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2!
model_5/tf.math.multiply_15/Mul?
!model_5/tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!model_5/tf.math.multiply_16/Mul/y?
model_5/tf.math.multiply_16/MulMul)model_5/max_pooling2d_16/MaxPool:output:0*model_5/tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2!
model_5/tf.math.multiply_16/Mul?
!model_5/tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2#
!model_5/tf.math.multiply_17/Mul/y?
model_5/tf.math.multiply_17/MulMul(model_5/activation_29/Relu:activations:0*model_5/tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2!
model_5/tf.math.multiply_17/Mul?
!model_5/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_5/concatenate_5/concat/axis?
model_5/concatenate_5/concatConcatV2#model_5/tf.math.multiply_15/Mul:z:0#model_5/tf.math.multiply_16/Mul:z:0#model_5/tf.math.multiply_17/Mul:z:0*model_5/concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
model_5/concatenate_5/concat?
)model_5/dense_39/MLCMatMul/ReadVariableOpReadVariableOp2model_5_dense_39_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02+
)model_5/dense_39/MLCMatMul/ReadVariableOp?
model_5/dense_39/MLCMatMul	MLCMatMul%model_5/concatenate_5/concat:output:01model_5/dense_39/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
model_5/dense_39/MLCMatMul?
'model_5/dense_39/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_39_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'model_5/dense_39/BiasAdd/ReadVariableOp?
model_5/dense_39/BiasAddBiasAdd$model_5/dense_39/MLCMatMul:product:0/model_5/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
model_5/dense_39/BiasAdd?
model_5/dense_39/TanhTanh!model_5/dense_39/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
model_5/dense_39/Tanh?
 model_5/max_pooling2d_17/MaxPoolMaxPoolmodel_5/dense_39/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2"
 model_5/max_pooling2d_17/MaxPool?
&model_5/tf.compat.v1.squeeze_5/SqueezeSqueeze)model_5/max_pooling2d_17/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2(
&model_5/tf.compat.v1.squeeze_5/Squeeze?
model_5/lstm_22/ShapeShape/model_5/tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:2
model_5/lstm_22/Shape?
#model_5/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_5/lstm_22/strided_slice/stack?
%model_5/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/lstm_22/strided_slice/stack_1?
%model_5/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/lstm_22/strided_slice/stack_2?
model_5/lstm_22/strided_sliceStridedSlicemodel_5/lstm_22/Shape:output:0,model_5/lstm_22/strided_slice/stack:output:0.model_5/lstm_22/strided_slice/stack_1:output:0.model_5/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_5/lstm_22/strided_slice}
model_5/lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_5/lstm_22/zeros/mul/y?
model_5/lstm_22/zeros/mulMul&model_5/lstm_22/strided_slice:output:0$model_5/lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_22/zeros/mul
model_5/lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_5/lstm_22/zeros/Less/y?
model_5/lstm_22/zeros/LessLessmodel_5/lstm_22/zeros/mul:z:0%model_5/lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_22/zeros/Less?
model_5/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
model_5/lstm_22/zeros/packed/1?
model_5/lstm_22/zeros/packedPack&model_5/lstm_22/strided_slice:output:0'model_5/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_5/lstm_22/zeros/packed
model_5/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_5/lstm_22/zeros/Const?
model_5/lstm_22/zerosFill%model_5/lstm_22/zeros/packed:output:0$model_5/lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_5/lstm_22/zeros?
model_5/lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_5/lstm_22/zeros_1/mul/y?
model_5/lstm_22/zeros_1/mulMul&model_5/lstm_22/strided_slice:output:0&model_5/lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_22/zeros_1/mul?
model_5/lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
model_5/lstm_22/zeros_1/Less/y?
model_5/lstm_22/zeros_1/LessLessmodel_5/lstm_22/zeros_1/mul:z:0'model_5/lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_22/zeros_1/Less?
 model_5/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2"
 model_5/lstm_22/zeros_1/packed/1?
model_5/lstm_22/zeros_1/packedPack&model_5/lstm_22/strided_slice:output:0)model_5/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_5/lstm_22/zeros_1/packed?
model_5/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_5/lstm_22/zeros_1/Const?
model_5/lstm_22/zeros_1Fill'model_5/lstm_22/zeros_1/packed:output:0&model_5/lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_5/lstm_22/zeros_1?
#model_5/lstm_22/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_5/lstm_22/MLCLSTM/hidden_size?
#model_5/lstm_22/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_5/lstm_22/MLCLSTM/output_size?
&model_5/lstm_22/MLCLSTM/ReadVariableOpReadVariableOp/model_5_lstm_22_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02(
&model_5/lstm_22/MLCLSTM/ReadVariableOp?
(model_5/lstm_22/MLCLSTM/ReadVariableOp_1ReadVariableOp1model_5_lstm_22_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02*
(model_5/lstm_22/MLCLSTM/ReadVariableOp_1?
(model_5/lstm_22/MLCLSTM/ReadVariableOp_2ReadVariableOp1model_5_lstm_22_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02*
(model_5/lstm_22/MLCLSTM/ReadVariableOp_2?
model_5/lstm_22/MLCLSTMMLCLSTM/model_5/tf.compat.v1.squeeze_5/Squeeze:output:0,model_5/lstm_22/MLCLSTM/hidden_size:output:0,model_5/lstm_22/MLCLSTM/output_size:output:0.model_5/lstm_22/MLCLSTM/ReadVariableOp:value:00model_5/lstm_22/MLCLSTM/ReadVariableOp_1:value:00model_5/lstm_22/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
model_5/lstm_22/MLCLSTM?
)model_5/dense_40/MLCMatMul/ReadVariableOpReadVariableOp2model_5_dense_40_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)model_5/dense_40/MLCMatMul/ReadVariableOp?
model_5/dense_40/MLCMatMul	MLCMatMul model_5/lstm_22/MLCLSTM:output:01model_5/dense_40/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
model_5/dense_40/MLCMatMul?
'model_5/dense_40/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_5/dense_40/BiasAdd/ReadVariableOp?
model_5/dense_40/BiasAddBiasAdd$model_5/dense_40/MLCMatMul:product:0/model_5/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_5/dense_40/BiasAdd?
model_5/dense_40/TanhTanh!model_5/dense_40/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_5/dense_40/Tanhw
model_5/lstm_23/ShapeShapemodel_5/dense_40/Tanh:y:0*
T0*
_output_shapes
:2
model_5/lstm_23/Shape?
#model_5/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_5/lstm_23/strided_slice/stack?
%model_5/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/lstm_23/strided_slice/stack_1?
%model_5/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/lstm_23/strided_slice/stack_2?
model_5/lstm_23/strided_sliceStridedSlicemodel_5/lstm_23/Shape:output:0,model_5/lstm_23/strided_slice/stack:output:0.model_5/lstm_23/strided_slice/stack_1:output:0.model_5/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_5/lstm_23/strided_slice|
model_5/lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model_5/lstm_23/zeros/mul/y?
model_5/lstm_23/zeros/mulMul&model_5/lstm_23/strided_slice:output:0$model_5/lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_23/zeros/mul
model_5/lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_5/lstm_23/zeros/Less/y?
model_5/lstm_23/zeros/LessLessmodel_5/lstm_23/zeros/mul:z:0%model_5/lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_23/zeros/Less?
model_5/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2 
model_5/lstm_23/zeros/packed/1?
model_5/lstm_23/zeros/packedPack&model_5/lstm_23/strided_slice:output:0'model_5/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_5/lstm_23/zeros/packed
model_5/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_5/lstm_23/zeros/Const?
model_5/lstm_23/zerosFill%model_5/lstm_23/zeros/packed:output:0$model_5/lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
model_5/lstm_23/zeros?
model_5/lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model_5/lstm_23/zeros_1/mul/y?
model_5/lstm_23/zeros_1/mulMul&model_5/lstm_23/strided_slice:output:0&model_5/lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_23/zeros_1/mul?
model_5/lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2 
model_5/lstm_23/zeros_1/Less/y?
model_5/lstm_23/zeros_1/LessLessmodel_5/lstm_23/zeros_1/mul:z:0'model_5/lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_5/lstm_23/zeros_1/Less?
 model_5/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2"
 model_5/lstm_23/zeros_1/packed/1?
model_5/lstm_23/zeros_1/packedPack&model_5/lstm_23/strided_slice:output:0)model_5/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
model_5/lstm_23/zeros_1/packed?
model_5/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_5/lstm_23/zeros_1/Const?
model_5/lstm_23/zeros_1Fill'model_5/lstm_23/zeros_1/packed:output:0&model_5/lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
model_5/lstm_23/zeros_1?
#model_5/lstm_23/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2%
#model_5/lstm_23/MLCLSTM/hidden_size?
#model_5/lstm_23/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2%
#model_5/lstm_23/MLCLSTM/output_size?
&model_5/lstm_23/MLCLSTM/ReadVariableOpReadVariableOp/model_5_lstm_23_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&model_5/lstm_23/MLCLSTM/ReadVariableOp?
(model_5/lstm_23/MLCLSTM/ReadVariableOp_1ReadVariableOp1model_5_lstm_23_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02*
(model_5/lstm_23/MLCLSTM/ReadVariableOp_1?
(model_5/lstm_23/MLCLSTM/ReadVariableOp_2ReadVariableOp1model_5_lstm_23_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02*
(model_5/lstm_23/MLCLSTM/ReadVariableOp_2?
model_5/lstm_23/MLCLSTMMLCLSTMmodel_5/dense_40/Tanh:y:0,model_5/lstm_23/MLCLSTM/hidden_size:output:0,model_5/lstm_23/MLCLSTM/output_size:output:0.model_5/lstm_23/MLCLSTM/ReadVariableOp:value:00model_5/lstm_23/MLCLSTM/ReadVariableOp_1:value:00model_5/lstm_23/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
model_5/lstm_23/MLCLSTM?
model_5/lstm_23/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
model_5/lstm_23/Reshape/shape?
model_5/lstm_23/ReshapeReshape model_5/lstm_23/MLCLSTM:output:0&model_5/lstm_23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
model_5/lstm_23/Reshape?
)model_5/dense_41/MLCMatMul/ReadVariableOpReadVariableOp2model_5_dense_41_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02+
)model_5/dense_41/MLCMatMul/ReadVariableOp?
model_5/dense_41/MLCMatMul	MLCMatMul model_5/lstm_23/Reshape:output:01model_5/dense_41/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model_5/dense_41/MLCMatMul?
'model_5/dense_41/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_41_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'model_5/dense_41/BiasAdd/ReadVariableOp?
model_5/dense_41/BiasAddBiasAdd$model_5/dense_41/MLCMatMul:product:0/model_5/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
model_5/dense_41/BiasAdd?
model_5/dense_41/TanhTanh!model_5/dense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
model_5/dense_41/Tanh?
IdentityIdentitymodel_5/dense_41/Tanh:y:0)^model_5/conv2d_15/BiasAdd/ReadVariableOp(^model_5/conv2d_15/Conv2D/ReadVariableOp)^model_5/conv2d_16/BiasAdd/ReadVariableOp(^model_5/conv2d_16/Conv2D/ReadVariableOp)^model_5/conv2d_17/BiasAdd/ReadVariableOp(^model_5/conv2d_17/Conv2D/ReadVariableOp(^model_5/dense_39/BiasAdd/ReadVariableOp*^model_5/dense_39/MLCMatMul/ReadVariableOp(^model_5/dense_40/BiasAdd/ReadVariableOp*^model_5/dense_40/MLCMatMul/ReadVariableOp(^model_5/dense_41/BiasAdd/ReadVariableOp*^model_5/dense_41/MLCMatMul/ReadVariableOp'^model_5/lstm_22/MLCLSTM/ReadVariableOp)^model_5/lstm_22/MLCLSTM/ReadVariableOp_1)^model_5/lstm_22/MLCLSTM/ReadVariableOp_2'^model_5/lstm_23/MLCLSTM/ReadVariableOp)^model_5/lstm_23/MLCLSTM/ReadVariableOp_1)^model_5/lstm_23/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2T
(model_5/conv2d_15/BiasAdd/ReadVariableOp(model_5/conv2d_15/BiasAdd/ReadVariableOp2R
'model_5/conv2d_15/Conv2D/ReadVariableOp'model_5/conv2d_15/Conv2D/ReadVariableOp2T
(model_5/conv2d_16/BiasAdd/ReadVariableOp(model_5/conv2d_16/BiasAdd/ReadVariableOp2R
'model_5/conv2d_16/Conv2D/ReadVariableOp'model_5/conv2d_16/Conv2D/ReadVariableOp2T
(model_5/conv2d_17/BiasAdd/ReadVariableOp(model_5/conv2d_17/BiasAdd/ReadVariableOp2R
'model_5/conv2d_17/Conv2D/ReadVariableOp'model_5/conv2d_17/Conv2D/ReadVariableOp2R
'model_5/dense_39/BiasAdd/ReadVariableOp'model_5/dense_39/BiasAdd/ReadVariableOp2V
)model_5/dense_39/MLCMatMul/ReadVariableOp)model_5/dense_39/MLCMatMul/ReadVariableOp2R
'model_5/dense_40/BiasAdd/ReadVariableOp'model_5/dense_40/BiasAdd/ReadVariableOp2V
)model_5/dense_40/MLCMatMul/ReadVariableOp)model_5/dense_40/MLCMatMul/ReadVariableOp2R
'model_5/dense_41/BiasAdd/ReadVariableOp'model_5/dense_41/BiasAdd/ReadVariableOp2V
)model_5/dense_41/MLCMatMul/ReadVariableOp)model_5/dense_41/MLCMatMul/ReadVariableOp2P
&model_5/lstm_22/MLCLSTM/ReadVariableOp&model_5/lstm_22/MLCLSTM/ReadVariableOp2T
(model_5/lstm_22/MLCLSTM/ReadVariableOp_1(model_5/lstm_22/MLCLSTM/ReadVariableOp_12T
(model_5/lstm_22/MLCLSTM/ReadVariableOp_2(model_5/lstm_22/MLCLSTM/ReadVariableOp_22P
&model_5/lstm_23/MLCLSTM/ReadVariableOp&model_5/lstm_23/MLCLSTM/ReadVariableOp2T
(model_5/lstm_23/MLCLSTM/ReadVariableOp_1(model_5/lstm_23/MLCLSTM/ReadVariableOp_12T
(model_5/lstm_23/MLCLSTM/ReadVariableOp_2(model_5/lstm_23/MLCLSTM/ReadVariableOp_2:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?

*__inference_conv2d_15_layer_call_fn_279444

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2783742
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_279520
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
?

?
D__inference_dense_39_layer_call_and_return_conditional_losses_279538

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

?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_279435

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
?
?
I__inference_concatenate_5_layer_call_and_return_conditional_losses_278471

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

*__inference_conv2d_17_layer_call_fn_279502

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2784002
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
D__inference_dense_40_layer_call_and_return_conditional_losses_279738

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
?
M
1__inference_max_pooling2d_16_layer_call_fn_277992

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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_2779862
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_278276

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
(__inference_lstm_23_layer_call_fn_279935
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2783232
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_278705

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
?L
?
C__inference_model_5_layer_call_and_return_conditional_losses_279003

inputs
inputs_1
inputs_2
conv2d_16_278944
conv2d_16_278946
conv2d_15_278949
conv2d_15_278951
conv2d_17_278954
conv2d_17_278956
dense_39_278971
dense_39_278973
lstm_22_278978
lstm_22_278980
lstm_22_278982
dense_40_278985
dense_40_278987
lstm_23_278990
lstm_23_278992
lstm_23_278994
dense_41_278997
dense_41_278999
identity??!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?lstm_22/StatefulPartitionedCall?lstm_23/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_16_278944conv2d_16_278946*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2783482#
!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15_278949conv2d_15_278951*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2783742#
!conv2d_15/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_17_278954conv2d_17_278956*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2784002#
!conv2d_17/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_2784212
activation_28/PartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_27_layer_call_and_return_conditional_losses_2784342
activation_27/PartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_2784472
activation_29/PartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_2779862"
 max_pooling2d_16/PartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2779742"
 max_pooling2d_15/PartitionedCall{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul)max_pooling2d_15/PartitionedCall:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul)max_pooling2d_16/PartitionedCall:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul&activation_29/PartitionedCall:output:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mul?
concatenate_5/PartitionedCallPartitionedCalltf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0*
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
GPU 2J 8? *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_2784712
concatenate_5/PartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_278971dense_39_278973*
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
GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2784922"
 dense_39/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_2779982"
 max_pooling2d_17/PartitionedCall?
tf.compat.v1.squeeze_5/SqueezeSqueeze)max_pooling2d_17/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeeze?
lstm_22/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_5/Squeeze:output:0lstm_22_278978lstm_22_278980lstm_22_278982*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785772!
lstm_22/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_40_278985dense_40_278987*
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
GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2786182"
 dense_40/StatefulPartitionedCall?
lstm_23/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_23_278990lstm_23_278992lstm_23_278994*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2787052!
lstm_23/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0dense_41_278997dense_41_278999*
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
GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_2787462"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:X T
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
?!
?
C__inference_lstm_22_layer_call_and_return_conditional_losses_278155

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
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_279339
inputs_0
inputs_1
inputs_2,
(conv2d_16_conv2d_readvariableop_resource-
)conv2d_16_biasadd_readvariableop_resource,
(conv2d_15_conv2d_readvariableop_resource-
)conv2d_15_biasadd_readvariableop_resource,
(conv2d_17_conv2d_readvariableop_resource-
)conv2d_17_biasadd_readvariableop_resource.
*dense_39_mlcmatmul_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource+
'lstm_22_mlclstm_readvariableop_resource-
)lstm_22_mlclstm_readvariableop_1_resource-
)lstm_22_mlclstm_readvariableop_2_resource.
*dense_40_mlcmatmul_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource+
'lstm_23_mlclstm_readvariableop_resource-
)lstm_23_mlclstm_readvariableop_1_resource-
)lstm_23_mlclstm_readvariableop_2_resource.
*dense_41_mlcmatmul_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity?? conv2d_15/BiasAdd/ReadVariableOp?conv2d_15/Conv2D/ReadVariableOp? conv2d_16/BiasAdd/ReadVariableOp?conv2d_16/Conv2D/ReadVariableOp? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp?dense_39/BiasAdd/ReadVariableOp?!dense_39/MLCMatMul/ReadVariableOp?dense_40/BiasAdd/ReadVariableOp?!dense_40/MLCMatMul/ReadVariableOp?dense_41/BiasAdd/ReadVariableOp?!dense_41/MLCMatMul/ReadVariableOp?lstm_22/MLCLSTM/ReadVariableOp? lstm_22/MLCLSTM/ReadVariableOp_1? lstm_22/MLCLSTM/ReadVariableOp_2?lstm_23/MLCLSTM/ReadVariableOp? lstm_23/MLCLSTM/ReadVariableOp_1? lstm_23/MLCLSTM/ReadVariableOp_2?
conv2d_16/Conv2D/ReadVariableOpReadVariableOp(conv2d_16_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_16/Conv2D/ReadVariableOp?
conv2d_16/Conv2D	MLCConv2Dinputs_1'conv2d_16/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_16/Conv2D?
 conv2d_16/BiasAdd/ReadVariableOpReadVariableOp)conv2d_16_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_16/BiasAdd/ReadVariableOp?
conv2d_16/BiasAddBiasAddconv2d_16/Conv2D:output:0(conv2d_16/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_16/BiasAdd?
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_15/Conv2D/ReadVariableOp?
conv2d_15/Conv2D	MLCConv2Dinputs_0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_15/Conv2D?
 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_15/BiasAdd/ReadVariableOp?
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_15/BiasAdd?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_17/Conv2D/ReadVariableOp?
conv2d_17/Conv2D	MLCConv2Dinputs_2'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_17/Conv2D?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_17/BiasAdd/ReadVariableOp?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_17/BiasAdd?
activation_28/ReluReluconv2d_16/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_28/Relu?
activation_27/ReluReluconv2d_15/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_27/Relu?
activation_29/ReluReluconv2d_17/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_29/Relu?
max_pooling2d_16/MaxPoolMaxPool activation_28/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_16/MaxPool?
max_pooling2d_15/MaxPoolMaxPool activation_27/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_15/MaxPool{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul!max_pooling2d_15/MaxPool:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul!max_pooling2d_16/MaxPool:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul activation_29/Relu:activations:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mulx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2tf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0"concatenate_5/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate_5/concat?
!dense_39/MLCMatMul/ReadVariableOpReadVariableOp*dense_39_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02#
!dense_39/MLCMatMul/ReadVariableOp?
dense_39/MLCMatMul	MLCMatMulconcatenate_5/concat:output:0)dense_39/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_39/MLCMatMul?
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_39/BiasAdd/ReadVariableOp?
dense_39/BiasAddBiasAdddense_39/MLCMatMul:product:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_39/BiasAdd|
dense_39/TanhTanhdense_39/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_39/Tanh?
max_pooling2d_17/MaxPoolMaxPooldense_39/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_17/MaxPool?
tf.compat.v1.squeeze_5/SqueezeSqueeze!max_pooling2d_17/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeezeu
lstm_22/ShapeShape'tf.compat.v1.squeeze_5/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_22/Shape?
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack?
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1?
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2?
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slicem
lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/mul/y?
lstm_22/zeros/mulMullstm_22/strided_slice:output:0lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/mulo
lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/Less/y?
lstm_22/zeros/LessLesslstm_22/zeros/mul:z:0lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/Lesss
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros/packed/1?
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const?
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/mul/y?
lstm_22/zeros_1/mulMullstm_22/strided_slice:output:0lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/muls
lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/Less/y?
lstm_22/zeros_1/LessLesslstm_22/zeros_1/mul:z:0lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/Lessw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/zeros_1/packed/1?
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const?
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_22/zeros_1}
lstm_22/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/MLCLSTM/hidden_size}
lstm_22/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_22/MLCLSTM/output_size?
lstm_22/MLCLSTM/ReadVariableOpReadVariableOp'lstm_22_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02 
lstm_22/MLCLSTM/ReadVariableOp?
 lstm_22/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_22_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02"
 lstm_22/MLCLSTM/ReadVariableOp_1?
 lstm_22/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_22_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_22/MLCLSTM/ReadVariableOp_2?
lstm_22/MLCLSTMMLCLSTM'tf.compat.v1.squeeze_5/Squeeze:output:0$lstm_22/MLCLSTM/hidden_size:output:0$lstm_22/MLCLSTM/output_size:output:0&lstm_22/MLCLSTM/ReadVariableOp:value:0(lstm_22/MLCLSTM/ReadVariableOp_1:value:0(lstm_22/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_22/MLCLSTM?
!dense_40/MLCMatMul/ReadVariableOpReadVariableOp*dense_40_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_40/MLCMatMul/ReadVariableOp?
dense_40/MLCMatMul	MLCMatMullstm_22/MLCLSTM:output:0)dense_40/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_40/MLCMatMul?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/MLCMatMul:product:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_40/BiasAddx
dense_40/TanhTanhdense_40/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_40/Tanh_
lstm_23/ShapeShapedense_40/Tanh:y:0*
T0*
_output_shapes
:2
lstm_23/Shape?
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stack?
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1?
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2?
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicel
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros/mul/y?
lstm_23/zeros/mulMullstm_23/strided_slice:output:0lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/mulo
lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_23/zeros/Less/y?
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lessr
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros/packed/1?
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros/packedo
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros/Const?
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/zerosp
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros_1/mul/y?
lstm_23/zeros_1/mulMullstm_23/strided_slice:output:0lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/muls
lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_23/zeros_1/Less/y?
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessv
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/zeros_1/packed/1?
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros_1/packeds
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros_1/Const?
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/zeros_1|
lstm_23/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/MLCLSTM/hidden_size|
lstm_23/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_23/MLCLSTM/output_size?
lstm_23/MLCLSTM/ReadVariableOpReadVariableOp'lstm_23_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02 
lstm_23/MLCLSTM/ReadVariableOp?
 lstm_23/MLCLSTM/ReadVariableOp_1ReadVariableOp)lstm_23_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02"
 lstm_23/MLCLSTM/ReadVariableOp_1?
 lstm_23/MLCLSTM/ReadVariableOp_2ReadVariableOp)lstm_23_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02"
 lstm_23/MLCLSTM/ReadVariableOp_2?
lstm_23/MLCLSTMMLCLSTMdense_40/Tanh:y:0$lstm_23/MLCLSTM/hidden_size:output:0$lstm_23/MLCLSTM/output_size:output:0&lstm_23/MLCLSTM/ReadVariableOp:value:0(lstm_23/MLCLSTM/ReadVariableOp_1:value:0(lstm_23/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_23/MLCLSTM
lstm_23/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_23/Reshape/shape?
lstm_23/ReshapeReshapelstm_23/MLCLSTM:output:0lstm_23/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_23/Reshape?
!dense_41/MLCMatMul/ReadVariableOpReadVariableOp*dense_41_mlcmatmul_readvariableop_resource*
_output_shapes

:<H*
dtype02#
!dense_41/MLCMatMul/ReadVariableOp?
dense_41/MLCMatMul	MLCMatMullstm_23/Reshape:output:0)dense_41/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_41/MLCMatMul?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/MLCMatMul:product:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_41/BiasAdds
dense_41/TanhTanhdense_41/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_41/Tanh?
IdentityIdentitydense_41/Tanh:y:0!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp!^conv2d_16/BiasAdd/ReadVariableOp ^conv2d_16/Conv2D/ReadVariableOp!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp ^dense_39/BiasAdd/ReadVariableOp"^dense_39/MLCMatMul/ReadVariableOp ^dense_40/BiasAdd/ReadVariableOp"^dense_40/MLCMatMul/ReadVariableOp ^dense_41/BiasAdd/ReadVariableOp"^dense_41/MLCMatMul/ReadVariableOp^lstm_22/MLCLSTM/ReadVariableOp!^lstm_22/MLCLSTM/ReadVariableOp_1!^lstm_22/MLCLSTM/ReadVariableOp_2^lstm_23/MLCLSTM/ReadVariableOp!^lstm_23/MLCLSTM/ReadVariableOp_1!^lstm_23/MLCLSTM/ReadVariableOp_2*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2D
 conv2d_16/BiasAdd/ReadVariableOp conv2d_16/BiasAdd/ReadVariableOp2B
conv2d_16/Conv2D/ReadVariableOpconv2d_16/Conv2D/ReadVariableOp2D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2B
dense_39/BiasAdd/ReadVariableOpdense_39/BiasAdd/ReadVariableOp2F
!dense_39/MLCMatMul/ReadVariableOp!dense_39/MLCMatMul/ReadVariableOp2B
dense_40/BiasAdd/ReadVariableOpdense_40/BiasAdd/ReadVariableOp2F
!dense_40/MLCMatMul/ReadVariableOp!dense_40/MLCMatMul/ReadVariableOp2B
dense_41/BiasAdd/ReadVariableOpdense_41/BiasAdd/ReadVariableOp2F
!dense_41/MLCMatMul/ReadVariableOp!dense_41/MLCMatMul/ReadVariableOp2@
lstm_22/MLCLSTM/ReadVariableOplstm_22/MLCLSTM/ReadVariableOp2D
 lstm_22/MLCLSTM/ReadVariableOp_1 lstm_22/MLCLSTM/ReadVariableOp_12D
 lstm_22/MLCLSTM/ReadVariableOp_2 lstm_22/MLCLSTM/ReadVariableOp_22@
lstm_23/MLCLSTM/ReadVariableOplstm_23/MLCLSTM/ReadVariableOp2D
 lstm_23/MLCLSTM/ReadVariableOp_1 lstm_23/MLCLSTM/ReadVariableOp_12D
 lstm_23/MLCLSTM/ReadVariableOp_2 lstm_23/MLCLSTM/ReadVariableOp_2:Z V
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

*__inference_conv2d_16_layer_call_fn_279463

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
GPU 2J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2783482
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
h
.__inference_concatenate_5_layer_call_fn_279527
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
GPU 2J 8? *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_2784712
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
?
?
$__inference_signature_wrapper_279095
conv2d_15_input
conv2d_16_input
conv2d_17_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_16_inputconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_2779682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?
?
(__inference_model_5_layer_call_fn_279042
conv2d_15_input
conv2d_16_input
conv2d_17_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_16_inputconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2790032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?
e
I__inference_activation_27_layer_call_and_return_conditional_losses_278434

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
J
.__inference_activation_27_layer_call_fn_279473

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
GPU 2J 8? *R
fMRK
I__inference_activation_27_layer_call_and_return_conditional_losses_2784342
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_278323

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
D__inference_dense_41_layer_call_and_return_conditional_losses_279946

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
?

?
D__inference_dense_41_layer_call_and_return_conditional_losses_278746

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
(__inference_lstm_22_layer_call_fn_279626
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2781102
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
e
I__inference_activation_29_layer_call_and_return_conditional_losses_279507

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
J
.__inference_activation_29_layer_call_fn_279512

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
GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_2784472
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
?
C__inference_lstm_22_layer_call_and_return_conditional_losses_278577

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

?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_279454

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
?
h
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_277998

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

?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_279493

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
?
(__inference_model_5_layer_call_fn_279425
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
GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2790032
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
? 
?
C__inference_lstm_22_layer_call_and_return_conditional_losses_279705

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
??
?#
"__inference__traced_restore_280380
file_prefix%
!assignvariableop_conv2d_15_kernel%
!assignvariableop_1_conv2d_15_bias'
#assignvariableop_2_conv2d_16_kernel%
!assignvariableop_3_conv2d_16_bias'
#assignvariableop_4_conv2d_17_kernel%
!assignvariableop_5_conv2d_17_bias&
"assignvariableop_6_dense_39_kernel$
 assignvariableop_7_dense_39_bias&
"assignvariableop_8_dense_40_kernel$
 assignvariableop_9_dense_40_bias'
#assignvariableop_10_dense_41_kernel%
!assignvariableop_11_dense_41_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate3
/assignvariableop_17_lstm_22_lstm_cell_44_kernel=
9assignvariableop_18_lstm_22_lstm_cell_44_recurrent_kernel1
-assignvariableop_19_lstm_22_lstm_cell_44_bias3
/assignvariableop_20_lstm_23_lstm_cell_45_kernel=
9assignvariableop_21_lstm_23_lstm_cell_45_recurrent_kernel1
-assignvariableop_22_lstm_23_lstm_cell_45_bias
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1
assignvariableop_27_total_2
assignvariableop_28_count_2/
+assignvariableop_29_adam_conv2d_15_kernel_m-
)assignvariableop_30_adam_conv2d_15_bias_m/
+assignvariableop_31_adam_conv2d_16_kernel_m-
)assignvariableop_32_adam_conv2d_16_bias_m/
+assignvariableop_33_adam_conv2d_17_kernel_m-
)assignvariableop_34_adam_conv2d_17_bias_m.
*assignvariableop_35_adam_dense_39_kernel_m,
(assignvariableop_36_adam_dense_39_bias_m.
*assignvariableop_37_adam_dense_40_kernel_m,
(assignvariableop_38_adam_dense_40_bias_m.
*assignvariableop_39_adam_dense_41_kernel_m,
(assignvariableop_40_adam_dense_41_bias_m:
6assignvariableop_41_adam_lstm_22_lstm_cell_44_kernel_mD
@assignvariableop_42_adam_lstm_22_lstm_cell_44_recurrent_kernel_m8
4assignvariableop_43_adam_lstm_22_lstm_cell_44_bias_m:
6assignvariableop_44_adam_lstm_23_lstm_cell_45_kernel_mD
@assignvariableop_45_adam_lstm_23_lstm_cell_45_recurrent_kernel_m8
4assignvariableop_46_adam_lstm_23_lstm_cell_45_bias_m/
+assignvariableop_47_adam_conv2d_15_kernel_v-
)assignvariableop_48_adam_conv2d_15_bias_v/
+assignvariableop_49_adam_conv2d_16_kernel_v-
)assignvariableop_50_adam_conv2d_16_bias_v/
+assignvariableop_51_adam_conv2d_17_kernel_v-
)assignvariableop_52_adam_conv2d_17_bias_v.
*assignvariableop_53_adam_dense_39_kernel_v,
(assignvariableop_54_adam_dense_39_bias_v.
*assignvariableop_55_adam_dense_40_kernel_v,
(assignvariableop_56_adam_dense_40_bias_v.
*assignvariableop_57_adam_dense_41_kernel_v,
(assignvariableop_58_adam_dense_41_bias_v:
6assignvariableop_59_adam_lstm_22_lstm_cell_44_kernel_vD
@assignvariableop_60_adam_lstm_22_lstm_cell_44_recurrent_kernel_v8
4assignvariableop_61_adam_lstm_22_lstm_cell_44_bias_v:
6assignvariableop_62_adam_lstm_23_lstm_cell_45_kernel_vD
@assignvariableop_63_adam_lstm_23_lstm_cell_45_recurrent_kernel_v8
4assignvariableop_64_adam_lstm_23_lstm_cell_45_bias_v
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_39_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_39_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_40_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_40_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_41_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_41_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp/assignvariableop_17_lstm_22_lstm_cell_44_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_lstm_22_lstm_cell_44_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp-assignvariableop_19_lstm_22_lstm_cell_44_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_lstm_23_lstm_cell_45_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp9assignvariableop_21_lstm_23_lstm_cell_45_recurrent_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_lstm_23_lstm_cell_45_biasIdentity_22:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_15_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_15_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_16_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_16_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_17_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_17_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_39_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_39_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_40_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_40_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_41_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_41_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_lstm_22_lstm_cell_44_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp@assignvariableop_42_adam_lstm_22_lstm_cell_44_recurrent_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_lstm_22_lstm_cell_44_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_lstm_23_lstm_cell_45_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp@assignvariableop_45_adam_lstm_23_lstm_cell_45_recurrent_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_lstm_23_lstm_cell_45_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_15_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_15_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_16_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_16_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_17_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_17_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_39_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_39_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_40_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_40_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_41_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_41_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_lstm_22_lstm_cell_44_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp@assignvariableop_60_adam_lstm_22_lstm_cell_44_recurrent_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adam_lstm_22_lstm_cell_44_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_lstm_23_lstm_cell_45_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp@assignvariableop_63_adam_lstm_23_lstm_cell_45_recurrent_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_lstm_23_lstm_cell_45_bias_vIdentity_64:output:0"/device:CPU:0*
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
M
1__inference_max_pooling2d_17_layer_call_fn_278004

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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_2779982
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
?
C__inference_lstm_22_layer_call_and_return_conditional_losses_279671

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
(__inference_model_5_layer_call_fn_279382
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
GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2788962
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
e
I__inference_activation_28_layer_call_and_return_conditional_losses_279478

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
e
I__inference_activation_29_layer_call_and_return_conditional_losses_278447

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
D__inference_dense_39_layer_call_and_return_conditional_losses_278492

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
?
~
)__inference_dense_39_layer_call_fn_279547

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
GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2784922
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
?
?
(__inference_lstm_23_layer_call_fn_279924
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2782762
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
?
h
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_277974

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
(__inference_model_5_layer_call_fn_278935
conv2d_15_input
conv2d_16_input
conv2d_17_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_16_inputconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_2788962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?M
?
C__inference_model_5_layer_call_and_return_conditional_losses_278827
conv2d_15_input
conv2d_16_input
conv2d_17_input
conv2d_16_278768
conv2d_16_278770
conv2d_15_278773
conv2d_15_278775
conv2d_17_278778
conv2d_17_278780
dense_39_278795
dense_39_278797
lstm_22_278802
lstm_22_278804
lstm_22_278806
dense_40_278809
dense_40_278811
lstm_23_278814
lstm_23_278816
lstm_23_278818
dense_41_278821
dense_41_278823
identity??!conv2d_15/StatefulPartitionedCall?!conv2d_16/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall? dense_39/StatefulPartitionedCall? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?lstm_22/StatefulPartitionedCall?lstm_23/StatefulPartitionedCall?
!conv2d_16/StatefulPartitionedCallStatefulPartitionedCallconv2d_16_inputconv2d_16_278768conv2d_16_278770*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_16_layer_call_and_return_conditional_losses_2783482#
!conv2d_16/StatefulPartitionedCall?
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCallconv2d_15_inputconv2d_15_278773conv2d_15_278775*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_2783742#
!conv2d_15/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_278778conv2d_17_278780*
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
GPU 2J 8? *N
fIRG
E__inference_conv2d_17_layer_call_and_return_conditional_losses_2784002#
!conv2d_17/StatefulPartitionedCall?
activation_28/PartitionedCallPartitionedCall*conv2d_16/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_28_layer_call_and_return_conditional_losses_2784212
activation_28/PartitionedCall?
activation_27/PartitionedCallPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_27_layer_call_and_return_conditional_losses_2784342
activation_27/PartitionedCall?
activation_29/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *R
fMRK
I__inference_activation_29_layer_call_and_return_conditional_losses_2784472
activation_29/PartitionedCall?
 max_pooling2d_16/PartitionedCallPartitionedCall&activation_28/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_2779862"
 max_pooling2d_16/PartitionedCall?
 max_pooling2d_15/PartitionedCallPartitionedCall&activation_27/PartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2779742"
 max_pooling2d_15/PartitionedCall{
tf.math.multiply_15/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_15/Mul/y?
tf.math.multiply_15/MulMul)max_pooling2d_15/PartitionedCall:output:0"tf.math.multiply_15/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_15/Mul{
tf.math.multiply_16/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_16/Mul/y?
tf.math.multiply_16/MulMul)max_pooling2d_16/PartitionedCall:output:0"tf.math.multiply_16/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_16/Mul{
tf.math.multiply_17/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_17/Mul/y?
tf.math.multiply_17/MulMul&activation_29/PartitionedCall:output:0"tf.math.multiply_17/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_17/Mul?
concatenate_5/PartitionedCallPartitionedCalltf.math.multiply_15/Mul:z:0tf.math.multiply_16/Mul:z:0tf.math.multiply_17/Mul:z:0*
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
GPU 2J 8? *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_2784712
concatenate_5/PartitionedCall?
 dense_39/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0dense_39_278795dense_39_278797*
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
GPU 2J 8? *M
fHRF
D__inference_dense_39_layer_call_and_return_conditional_losses_2784922"
 dense_39/StatefulPartitionedCall?
 max_pooling2d_17/PartitionedCallPartitionedCall)dense_39/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_2779982"
 max_pooling2d_17/PartitionedCall?
tf.compat.v1.squeeze_5/SqueezeSqueeze)max_pooling2d_17/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_5/Squeeze?
lstm_22/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_5/Squeeze:output:0lstm_22_278802lstm_22_278804lstm_22_278806*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_2785772!
lstm_22/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_40_278809dense_40_278811*
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
GPU 2J 8? *M
fHRF
D__inference_dense_40_layer_call_and_return_conditional_losses_2786182"
 dense_40/StatefulPartitionedCall?
lstm_23/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0lstm_23_278814lstm_23_278816lstm_23_278818*
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
GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2787052!
lstm_23/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0dense_41_278821dense_41_278823*
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
GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_2787462"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0"^conv2d_15/StatefulPartitionedCall"^conv2d_16/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2F
!conv2d_16/StatefulPartitionedCall!conv2d_16/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_15_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_16_input:a]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_17_input
?
e
I__inference_activation_28_layer_call_and_return_conditional_losses_278421

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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_278348

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
?
M
1__inference_max_pooling2d_15_layer_call_fn_277980

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
GPU 2J 8? *U
fPRN
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_2779742
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
?
h
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_277986

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
(__inference_lstm_23_layer_call_fn_279841

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
 *'
_output_shapes
:?????????<*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_lstm_23_layer_call_and_return_conditional_losses_2787052
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

?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_278400

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
??
?
__inference__traced_save_280175
file_prefix/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop/
+savev2_conv2d_16_kernel_read_readvariableop-
)savev2_conv2d_16_bias_read_readvariableop/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_22_lstm_cell_44_kernel_read_readvariableopD
@savev2_lstm_22_lstm_cell_44_recurrent_kernel_read_readvariableop8
4savev2_lstm_22_lstm_cell_44_bias_read_readvariableop:
6savev2_lstm_23_lstm_cell_45_kernel_read_readvariableopD
@savev2_lstm_23_lstm_cell_45_recurrent_kernel_read_readvariableop8
4savev2_lstm_23_lstm_cell_45_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_conv2d_15_kernel_m_read_readvariableop4
0savev2_adam_conv2d_15_bias_m_read_readvariableop6
2savev2_adam_conv2d_16_kernel_m_read_readvariableop4
0savev2_adam_conv2d_16_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableop5
1savev2_adam_dense_39_kernel_m_read_readvariableop3
/savev2_adam_dense_39_bias_m_read_readvariableop5
1savev2_adam_dense_40_kernel_m_read_readvariableop3
/savev2_adam_dense_40_bias_m_read_readvariableop5
1savev2_adam_dense_41_kernel_m_read_readvariableop3
/savev2_adam_dense_41_bias_m_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_44_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_44_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_44_bias_m_read_readvariableopA
=savev2_adam_lstm_23_lstm_cell_45_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_23_lstm_cell_45_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_23_lstm_cell_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_15_kernel_v_read_readvariableop4
0savev2_adam_conv2d_15_bias_v_read_readvariableop6
2savev2_adam_conv2d_16_kernel_v_read_readvariableop4
0savev2_adam_conv2d_16_bias_v_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableop5
1savev2_adam_dense_39_kernel_v_read_readvariableop3
/savev2_adam_dense_39_bias_v_read_readvariableop5
1savev2_adam_dense_40_kernel_v_read_readvariableop3
/savev2_adam_dense_40_bias_v_read_readvariableop5
1savev2_adam_dense_41_kernel_v_read_readvariableop3
/savev2_adam_dense_41_bias_v_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_44_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_44_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_44_bias_v_read_readvariableopA
=savev2_adam_lstm_23_lstm_cell_45_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_23_lstm_cell_45_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_23_lstm_cell_45_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop+savev2_conv2d_16_kernel_read_readvariableop)savev2_conv2d_16_bias_read_readvariableop+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_22_lstm_cell_44_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_44_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_44_bias_read_readvariableop6savev2_lstm_23_lstm_cell_45_kernel_read_readvariableop@savev2_lstm_23_lstm_cell_45_recurrent_kernel_read_readvariableop4savev2_lstm_23_lstm_cell_45_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_conv2d_15_kernel_m_read_readvariableop0savev2_adam_conv2d_15_bias_m_read_readvariableop2savev2_adam_conv2d_16_kernel_m_read_readvariableop0savev2_adam_conv2d_16_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop1savev2_adam_dense_39_kernel_m_read_readvariableop/savev2_adam_dense_39_bias_m_read_readvariableop1savev2_adam_dense_40_kernel_m_read_readvariableop/savev2_adam_dense_40_bias_m_read_readvariableop1savev2_adam_dense_41_kernel_m_read_readvariableop/savev2_adam_dense_41_bias_m_read_readvariableop=savev2_adam_lstm_22_lstm_cell_44_kernel_m_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_44_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_22_lstm_cell_44_bias_m_read_readvariableop=savev2_adam_lstm_23_lstm_cell_45_kernel_m_read_readvariableopGsavev2_adam_lstm_23_lstm_cell_45_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_23_lstm_cell_45_bias_m_read_readvariableop2savev2_adam_conv2d_15_kernel_v_read_readvariableop0savev2_adam_conv2d_15_bias_v_read_readvariableop2savev2_adam_conv2d_16_kernel_v_read_readvariableop0savev2_adam_conv2d_16_bias_v_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop1savev2_adam_dense_39_kernel_v_read_readvariableop/savev2_adam_dense_39_bias_v_read_readvariableop1savev2_adam_dense_40_kernel_v_read_readvariableop/savev2_adam_dense_40_bias_v_read_readvariableop1savev2_adam_dense_41_kernel_v_read_readvariableop/savev2_adam_dense_41_bias_v_read_readvariableop=savev2_adam_lstm_22_lstm_cell_44_kernel_v_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_44_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_22_lstm_cell_44_bias_v_read_readvariableop=savev2_adam_lstm_23_lstm_cell_45_kernel_v_read_readvariableopGsavev2_adam_lstm_23_lstm_cell_45_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_23_lstm_cell_45_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?"
?
C__inference_lstm_23_layer_call_and_return_conditional_losses_279783

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
?"
?
C__inference_lstm_23_layer_call_and_return_conditional_losses_279819

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
?
~
)__inference_dense_41_layer_call_fn_279955

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
GPU 2J 8? *M
fHRF
D__inference_dense_41_layer_call_and_return_conditional_losses_2787462
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
T
conv2d_15_inputA
!serving_default_conv2d_15_input:0??????????
T
conv2d_16_inputA
!serving_default_conv2d_16_input:0??????????
T
conv2d_17_inputA
!serving_default_conv2d_17_input:0??????????<
dense_410
StatefulPartitionedCall:0?????????Htensorflow/serving/predict:??
ݪ
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_15_input"}, "name": "conv2d_15_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_16_input"}, "name": "conv2d_16_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["conv2d_15_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_16_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_17_input"}, "name": "conv2d_17_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_27", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_27", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_28", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_28", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["conv2d_17_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_15", "inbound_nodes": [[["activation_27", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_16", "inbound_nodes": [[["activation_28", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_29", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_29", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["max_pooling2d_15", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_16", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_16", "inbound_nodes": [["max_pooling2d_16", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_17", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_17", "inbound_nodes": [["activation_29", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["tf.math.multiply_15", 0, 0, {}], ["tf.math.multiply_16", 0, 0, {}], ["tf.math.multiply_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_17", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze_5", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze_5", "inbound_nodes": [["max_pooling2d_17", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_22", "inbound_nodes": [[["tf.compat.v1.squeeze_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["lstm_22", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_23", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_23", "inbound_nodes": [[["dense_40", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_41", "inbound_nodes": [[["lstm_23", 0, 0, {}]]]}], "input_layers": [["conv2d_15_input", 0, 0], ["conv2d_16_input", 0, 0], ["conv2d_17_input", 0, 0]], "output_layers": [["dense_41", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 144, 5]}, {"class_name": "TensorShape", "items": [null, 1, 144, 5]}, {"class_name": "TensorShape", "items": [null, 1, 144, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_15_input"}, "name": "conv2d_15_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_16_input"}, "name": "conv2d_16_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_15", "inbound_nodes": [[["conv2d_15_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_16", "inbound_nodes": [[["conv2d_16_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_17_input"}, "name": "conv2d_17_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_27", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_27", "inbound_nodes": [[["conv2d_15", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_28", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_28", "inbound_nodes": [[["conv2d_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_17", "inbound_nodes": [[["conv2d_17_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_15", "inbound_nodes": [[["activation_27", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_16", "inbound_nodes": [[["activation_28", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_29", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_29", "inbound_nodes": [[["conv2d_17", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_15", "inbound_nodes": [["max_pooling2d_15", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_16", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_16", "inbound_nodes": [["max_pooling2d_16", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_17", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_17", "inbound_nodes": [["activation_29", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["tf.math.multiply_15", 0, 0, {}], ["tf.math.multiply_16", 0, 0, {}], ["tf.math.multiply_17", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_39", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_17", "inbound_nodes": [[["dense_39", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze_5", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze_5", "inbound_nodes": [["max_pooling2d_17", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_22", "inbound_nodes": [[["tf.compat.v1.squeeze_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_40", "inbound_nodes": [[["lstm_22", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_23", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_23", "inbound_nodes": [[["dense_40", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_41", "inbound_nodes": [[["lstm_23", 0, 0, {}]]]}], "input_layers": [["conv2d_15_input", 0, 0], ["conv2d_16_input", 0, 0], ["conv2d_17_input", 0, 0]], "output_layers": [["dense_41", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_15_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_15_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_16_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_16_input"}}
?


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?


#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_16", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_17_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_17_input"}}
?
)regularization_losses
*	variables
+trainable_variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_27", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
-regularization_losses
.	variables
/trainable_variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_28", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_17", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 144, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 5]}}
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_15", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_16", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_29", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
C	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_15", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
D	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_16", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
E	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_17", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 144, 32]}, {"class_name": "TensorShape", "items": [null, 1, 144, 32]}, {"class_name": "TensorShape", "items": [null, 1, 144, 32]}]}
?

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 144, 96]}}
?
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
T	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.squeeze_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.squeeze_5", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}}
?
Ucell
V
state_spec
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "lstm_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 48]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 48]}}
?

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 200]}}
?
acell
b
state_spec
cregularization_losses
d	variables
etrainable_variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_23", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 30]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 144, 30]}}
?

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 72, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
?
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem?m?#m?$m?1m?2m?Jm?Km?[m?\m?gm?hm?rm?sm?tm?um?vm?wm?v?v?#v?$v?1v?2v?Jv?Kv?[v?\v?gv?hv?rv?sv?tv?uv?vv?wv?"
	optimizer
 "
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
?
regularization_losses
xlayer_regularization_losses
ylayer_metrics
	variables
znon_trainable_variables
trainable_variables
{metrics

|layers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_15/kernel
: 2conv2d_15/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
}layer_regularization_losses
~layer_metrics
 	variables
non_trainable_variables
!trainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_16/kernel
: 2conv2d_16/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
%regularization_losses
 ?layer_regularization_losses
?layer_metrics
&	variables
?non_trainable_variables
'trainable_variables
?metrics
?layers
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
)regularization_losses
 ?layer_regularization_losses
?layer_metrics
*	variables
?non_trainable_variables
+trainable_variables
?metrics
?layers
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
-regularization_losses
 ?layer_regularization_losses
?layer_metrics
.	variables
?non_trainable_variables
/trainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_17/kernel
: 2conv2d_17/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
3regularization_losses
 ?layer_regularization_losses
?layer_metrics
4	variables
?non_trainable_variables
5trainable_variables
?metrics
?layers
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
7regularization_losses
 ?layer_regularization_losses
?layer_metrics
8	variables
?non_trainable_variables
9trainable_variables
?metrics
?layers
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
;regularization_losses
 ?layer_regularization_losses
?layer_metrics
<	variables
?non_trainable_variables
=trainable_variables
?metrics
?layers
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
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
@	variables
?non_trainable_variables
Atrainable_variables
?metrics
?layers
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
Fregularization_losses
 ?layer_regularization_losses
?layer_metrics
G	variables
?non_trainable_variables
Htrainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:`02dense_39/kernel
:02dense_39/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
M	variables
?non_trainable_variables
Ntrainable_variables
?metrics
?layers
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
Pregularization_losses
 ?layer_regularization_losses
?layer_metrics
Q	variables
?non_trainable_variables
Rtrainable_variables
?metrics
?layers
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
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_44", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
?
Wregularization_losses
 ?layer_regularization_losses
?layer_metrics
X	variables
?non_trainable_variables
Ytrainable_variables
?metrics
?layers
?states
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_40/kernel
:2dense_40/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
]regularization_losses
 ?layer_regularization_losses
?layer_metrics
^	variables
?non_trainable_variables
_trainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

ukernel
vrecurrent_kernel
wbias
?regularization_losses
?	variables
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_45", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
?
cregularization_losses
 ?layer_regularization_losses
?layer_metrics
d	variables
?non_trainable_variables
etrainable_variables
?metrics
?layers
?states
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:<H2dense_41/kernel
:H2dense_41/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
?
iregularization_losses
 ?layer_regularization_losses
?layer_metrics
j	variables
?non_trainable_variables
ktrainable_variables
?metrics
?layers
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
.:,	0?2lstm_22/lstm_cell_44/kernel
9:7
??2%lstm_22/lstm_cell_44/recurrent_kernel
(:&?2lstm_22/lstm_cell_44/bias
.:,	?2lstm_23/lstm_cell_45/kernel
8:6	<?2%lstm_23/lstm_cell_45/recurrent_kernel
(:&?2lstm_23/lstm_cell_45/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
?
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
?
?regularization_losses
 ?layer_regularization_losses
?layer_metrics
?	variables
?non_trainable_variables
?trainable_variables
?metrics
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
/:- 2Adam/conv2d_15/kernel/m
!: 2Adam/conv2d_15/bias/m
/:- 2Adam/conv2d_16/kernel/m
!: 2Adam/conv2d_16/bias/m
/:- 2Adam/conv2d_17/kernel/m
!: 2Adam/conv2d_17/bias/m
&:$`02Adam/dense_39/kernel/m
 :02Adam/dense_39/bias/m
':%	?2Adam/dense_40/kernel/m
 :2Adam/dense_40/bias/m
&:$<H2Adam/dense_41/kernel/m
 :H2Adam/dense_41/bias/m
3:1	0?2"Adam/lstm_22/lstm_cell_44/kernel/m
>:<
??2,Adam/lstm_22/lstm_cell_44/recurrent_kernel/m
-:+?2 Adam/lstm_22/lstm_cell_44/bias/m
3:1	?2"Adam/lstm_23/lstm_cell_45/kernel/m
=:;	<?2,Adam/lstm_23/lstm_cell_45/recurrent_kernel/m
-:+?2 Adam/lstm_23/lstm_cell_45/bias/m
/:- 2Adam/conv2d_15/kernel/v
!: 2Adam/conv2d_15/bias/v
/:- 2Adam/conv2d_16/kernel/v
!: 2Adam/conv2d_16/bias/v
/:- 2Adam/conv2d_17/kernel/v
!: 2Adam/conv2d_17/bias/v
&:$`02Adam/dense_39/kernel/v
 :02Adam/dense_39/bias/v
':%	?2Adam/dense_40/kernel/v
 :2Adam/dense_40/bias/v
&:$<H2Adam/dense_41/kernel/v
 :H2Adam/dense_41/bias/v
3:1	0?2"Adam/lstm_22/lstm_cell_44/kernel/v
>:<
??2,Adam/lstm_22/lstm_cell_44/recurrent_kernel/v
-:+?2 Adam/lstm_22/lstm_cell_44/bias/v
3:1	?2"Adam/lstm_23/lstm_cell_45/kernel/v
=:;	<?2,Adam/lstm_23/lstm_cell_45/recurrent_kernel/v
-:+?2 Adam/lstm_23/lstm_cell_45/bias/v
?2?
(__inference_model_5_layer_call_fn_279382
(__inference_model_5_layer_call_fn_279042
(__inference_model_5_layer_call_fn_279425
(__inference_model_5_layer_call_fn_278935?
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
!__inference__wrapped_model_277968?
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
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
?2?
C__inference_model_5_layer_call_and_return_conditional_losses_278763
C__inference_model_5_layer_call_and_return_conditional_losses_279339
C__inference_model_5_layer_call_and_return_conditional_losses_279217
C__inference_model_5_layer_call_and_return_conditional_losses_278827?
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
*__inference_conv2d_15_layer_call_fn_279444?
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
E__inference_conv2d_15_layer_call_and_return_conditional_losses_279435?
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
*__inference_conv2d_16_layer_call_fn_279463?
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
E__inference_conv2d_16_layer_call_and_return_conditional_losses_279454?
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
.__inference_activation_27_layer_call_fn_279473?
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
I__inference_activation_27_layer_call_and_return_conditional_losses_279468?
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
.__inference_activation_28_layer_call_fn_279483?
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
I__inference_activation_28_layer_call_and_return_conditional_losses_279478?
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
*__inference_conv2d_17_layer_call_fn_279502?
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
E__inference_conv2d_17_layer_call_and_return_conditional_losses_279493?
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
1__inference_max_pooling2d_15_layer_call_fn_277980?
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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_277974?
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
1__inference_max_pooling2d_16_layer_call_fn_277992?
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
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_277986?
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
.__inference_activation_29_layer_call_fn_279512?
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
I__inference_activation_29_layer_call_and_return_conditional_losses_279507?
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
.__inference_concatenate_5_layer_call_fn_279527?
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
I__inference_concatenate_5_layer_call_and_return_conditional_losses_279520?
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
)__inference_dense_39_layer_call_fn_279547?
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
D__inference_dense_39_layer_call_and_return_conditional_losses_279538?
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
1__inference_max_pooling2d_17_layer_call_fn_278004?
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
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_277998?
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
(__inference_lstm_22_layer_call_fn_279626
(__inference_lstm_22_layer_call_fn_279637
(__inference_lstm_22_layer_call_fn_279716
(__inference_lstm_22_layer_call_fn_279727?
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_279615
C__inference_lstm_22_layer_call_and_return_conditional_losses_279705
C__inference_lstm_22_layer_call_and_return_conditional_losses_279581
C__inference_lstm_22_layer_call_and_return_conditional_losses_279671?
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
)__inference_dense_40_layer_call_fn_279747?
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
D__inference_dense_40_layer_call_and_return_conditional_losses_279738?
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
(__inference_lstm_23_layer_call_fn_279841
(__inference_lstm_23_layer_call_fn_279830
(__inference_lstm_23_layer_call_fn_279924
(__inference_lstm_23_layer_call_fn_279935?
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_279783
C__inference_lstm_23_layer_call_and_return_conditional_losses_279877
C__inference_lstm_23_layer_call_and_return_conditional_losses_279913
C__inference_lstm_23_layer_call_and_return_conditional_losses_279819?
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
)__inference_dense_41_layer_call_fn_279955?
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
D__inference_dense_41_layer_call_and_return_conditional_losses_279946?
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
$__inference_signature_wrapper_279095conv2d_15_inputconv2d_16_inputconv2d_17_input"?
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
!__inference__wrapped_model_277968?#$12JKrst[\uvwgh???
???
???
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
? "3?0
.
dense_41"?
dense_41?????????H?
I__inference_activation_27_layer_call_and_return_conditional_losses_279468j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
.__inference_activation_27_layer_call_fn_279473]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
I__inference_activation_28_layer_call_and_return_conditional_losses_279478j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
.__inference_activation_28_layer_call_fn_279483]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
I__inference_activation_29_layer_call_and_return_conditional_losses_279507j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
.__inference_activation_29_layer_call_fn_279512]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
I__inference_concatenate_5_layer_call_and_return_conditional_losses_279520????
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
.__inference_concatenate_5_layer_call_fn_279527????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? "!???????????`?
E__inference_conv2d_15_layer_call_and_return_conditional_losses_279435n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
*__inference_conv2d_15_layer_call_fn_279444a8?5
.?+
)?&
inputs??????????
? "!??????????? ?
E__inference_conv2d_16_layer_call_and_return_conditional_losses_279454n#$8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
*__inference_conv2d_16_layer_call_fn_279463a#$8?5
.?+
)?&
inputs??????????
? "!??????????? ?
E__inference_conv2d_17_layer_call_and_return_conditional_losses_279493n128?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
*__inference_conv2d_17_layer_call_fn_279502a128?5
.?+
)?&
inputs??????????
? "!??????????? ?
D__inference_dense_39_layer_call_and_return_conditional_losses_279538nJK8?5
.?+
)?&
inputs??????????`
? ".?+
$?!
0??????????0
? ?
)__inference_dense_39_layer_call_fn_279547aJK8?5
.?+
)?&
inputs??????????`
? "!???????????0?
D__inference_dense_40_layer_call_and_return_conditional_losses_279738g[\5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????
? ?
)__inference_dense_40_layer_call_fn_279747Z[\5?2
+?(
&?#
inputs???????????
? "????????????
D__inference_dense_41_layer_call_and_return_conditional_losses_279946\gh/?,
%?"
 ?
inputs?????????<
? "%?"
?
0?????????H
? |
)__inference_dense_41_layer_call_fn_279955Ogh/?,
%?"
 ?
inputs?????????<
? "??????????H?
C__inference_lstm_22_layer_call_and_return_conditional_losses_279581?rstO?L
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_279615?rstO?L
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_279671trst@?=
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
C__inference_lstm_22_layer_call_and_return_conditional_losses_279705trst@?=
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
(__inference_lstm_22_layer_call_fn_279626~rstO?L
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
(__inference_lstm_22_layer_call_fn_279637~rstO?L
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
(__inference_lstm_22_layer_call_fn_279716grst@?=
6?3
%?"
inputs??????????0

 
p

 
? "?????????????
(__inference_lstm_22_layer_call_fn_279727grst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "?????????????
C__inference_lstm_23_layer_call_and_return_conditional_losses_279783nuvw@?=
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_279819nuvw@?=
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_279877}uvwO?L
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
C__inference_lstm_23_layer_call_and_return_conditional_losses_279913}uvwO?L
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
(__inference_lstm_23_layer_call_fn_279830auvw@?=
6?3
%?"
inputs??????????

 
p

 
? "??????????<?
(__inference_lstm_23_layer_call_fn_279841auvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "??????????<?
(__inference_lstm_23_layer_call_fn_279924puvwO?L
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
(__inference_lstm_23_layer_call_fn_279935puvwO?L
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
L__inference_max_pooling2d_15_layer_call_and_return_conditional_losses_277974?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_15_layer_call_fn_277980?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_277986?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_16_layer_call_fn_277992?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_277998?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_17_layer_call_fn_278004?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_5_layer_call_and_return_conditional_losses_278763?#$12JKrst[\uvwgh???
???
???
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
p

 
? "%?"
?
0?????????H
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_278827?#$12JKrst[\uvwgh???
???
???
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
p 

 
? "%?"
?
0?????????H
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_279217?#$12JKrst[\uvwgh???
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
C__inference_model_5_layer_call_and_return_conditional_losses_279339?#$12JKrst[\uvwgh???
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
(__inference_model_5_layer_call_fn_278935?#$12JKrst[\uvwgh???
???
???
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
p

 
? "??????????H?
(__inference_model_5_layer_call_fn_279042?#$12JKrst[\uvwgh???
???
???
2?/
conv2d_15_input??????????
2?/
conv2d_16_input??????????
2?/
conv2d_17_input??????????
p 

 
? "??????????H?
(__inference_model_5_layer_call_fn_279382?#$12JKrst[\uvwgh???
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
(__inference_model_5_layer_call_fn_279425?#$12JKrst[\uvwgh???
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
$__inference_signature_wrapper_279095?#$12JKrst[\uvwgh???
? 
???
E
conv2d_15_input2?/
conv2d_15_input??????????
E
conv2d_16_input2?/
conv2d_16_input??????????
E
conv2d_17_input2?/
conv2d_17_input??????????"3?0
.
dense_41"?
dense_41?????????H