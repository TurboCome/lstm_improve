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
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
: *
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
: *
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:`0*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:0*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	?*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	<?*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
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
lstm_6/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*+
shared_namelstm_6/lstm_cell_12/kernel
?
.lstm_6/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_12/kernel*
_output_shapes
:	0?*
dtype0
?
$lstm_6/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*5
shared_name&$lstm_6/lstm_cell_12/recurrent_kernel
?
8lstm_6/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_6/lstm_cell_12/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_6/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelstm_6/lstm_cell_12/bias
?
,lstm_6/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_6/lstm_cell_12/bias*
_output_shapes	
:?*
dtype0
?
lstm_7/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_namelstm_7/lstm_cell_13/kernel
?
.lstm_7/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_7/lstm_cell_13/kernel*
_output_shapes
:	?*
dtype0
?
$lstm_7/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*5
shared_name&$lstm_7/lstm_cell_13/recurrent_kernel
?
8lstm_7/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp$lstm_7/lstm_cell_13/recurrent_kernel*
_output_shapes
:	<?*
dtype0
?
lstm_7/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namelstm_7/lstm_cell_13/bias
?
,lstm_7/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_7/lstm_cell_13/bias*
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
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_4/kernel/m
?
*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/m
?
*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:`0*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/m
?
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*'
shared_nameAdam/dense_11/kernel/m
?
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	<?*
dtype0
?
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_11/bias/m
z
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_6/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*2
shared_name#!Adam/lstm_6/lstm_cell_12/kernel/m
?
5Adam/lstm_6/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_6/lstm_cell_12/kernel/m*
_output_shapes
:	0?*
dtype0
?
+Adam/lstm_6/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*<
shared_name-+Adam/lstm_6/lstm_cell_12/recurrent_kernel/m
?
?Adam/lstm_6/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_6/lstm_cell_12/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_6/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_6/lstm_cell_12/bias/m
?
3Adam/lstm_6/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_12/bias/m*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_7/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/lstm_7/lstm_cell_13/kernel/m
?
5Adam/lstm_7/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/lstm_7/lstm_cell_13/kernel/m*
_output_shapes
:	?*
dtype0
?
+Adam/lstm_7/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*<
shared_name-+Adam/lstm_7/lstm_cell_13/recurrent_kernel/m
?
?Adam/lstm_7/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp+Adam/lstm_7/lstm_cell_13/recurrent_kernel/m*
_output_shapes
:	<?*
dtype0
?
Adam/lstm_7/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_7/lstm_cell_13/bias/m
?
3Adam/lstm_7/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_7/lstm_cell_13/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_4/kernel/v
?
*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_5/kernel/v
?
*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`0*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:`0*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_10/kernel/v
?
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*'
shared_nameAdam/dense_11/kernel/v
?
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	<?*
dtype0
?
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_11/bias/v
z
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_6/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	0?*2
shared_name#!Adam/lstm_6/lstm_cell_12/kernel/v
?
5Adam/lstm_6/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_6/lstm_cell_12/kernel/v*
_output_shapes
:	0?*
dtype0
?
+Adam/lstm_6/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*<
shared_name-+Adam/lstm_6/lstm_cell_12/recurrent_kernel/v
?
?Adam/lstm_6/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_6/lstm_cell_12/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/lstm_6/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_6/lstm_cell_12/bias/v
?
3Adam/lstm_6/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_6/lstm_cell_12/bias/v*
_output_shapes	
:?*
dtype0
?
!Adam/lstm_7/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/lstm_7/lstm_cell_13/kernel/v
?
5Adam/lstm_7/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/lstm_7/lstm_cell_13/kernel/v*
_output_shapes
:	?*
dtype0
?
+Adam/lstm_7/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	<?*<
shared_name-+Adam/lstm_7/lstm_cell_13/recurrent_kernel/v
?
?Adam/lstm_7/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp+Adam/lstm_7/lstm_cell_13/recurrent_kernel/v*
_output_shapes
:	<?*
dtype0
?
Adam/lstm_7/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!Adam/lstm_7/lstm_cell_13/bias/v
?
3Adam/lstm_7/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_7/lstm_cell_13/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?r
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
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
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
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VT
VARIABLE_VALUElstm_6/lstm_cell_12/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$lstm_6/lstm_cell_12/recurrent_kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_6/lstm_cell_12/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_7/lstm_cell_13/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE$lstm_7/lstm_cell_13/recurrent_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_7/lstm_cell_13/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
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
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_6/lstm_cell_12/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_6/lstm_cell_12/recurrent_kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_6/lstm_cell_12/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/lstm_7/lstm_cell_13/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_7/lstm_cell_13/recurrent_kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_7/lstm_cell_13/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/lstm_6/lstm_cell_12/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_6/lstm_cell_12/recurrent_kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_6/lstm_cell_12/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/lstm_7/lstm_cell_13/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/lstm_7/lstm_cell_13/recurrent_kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/lstm_7/lstm_cell_13/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_3_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_4_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
serving_default_conv2d_5_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_3_inputserving_default_conv2d_4_inputserving_default_conv2d_5_inputconv2d_4/kernelconv2d_4/biasconv2d_3/kernelconv2d_3/biasconv2d_5/kernelconv2d_5/biasdense_9/kerneldense_9/biaslstm_6/lstm_cell_12/kernel$lstm_6/lstm_cell_12/recurrent_kernellstm_6/lstm_cell_12/biasdense_10/kerneldense_10/biaslstm_7/lstm_cell_13/kernel$lstm_7/lstm_cell_13/recurrent_kernellstm_7/lstm_cell_13/biasdense_11/kerneldense_11/bias* 
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
#__inference_signature_wrapper_72456
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp.lstm_6/lstm_cell_12/kernel/Read/ReadVariableOp8lstm_6/lstm_cell_12/recurrent_kernel/Read/ReadVariableOp,lstm_6/lstm_cell_12/bias/Read/ReadVariableOp.lstm_7/lstm_cell_13/kernel/Read/ReadVariableOp8lstm_7/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp,lstm_7/lstm_cell_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOp5Adam/lstm_6/lstm_cell_12/kernel/m/Read/ReadVariableOp?Adam/lstm_6/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_6/lstm_cell_12/bias/m/Read/ReadVariableOp5Adam/lstm_7/lstm_cell_13/kernel/m/Read/ReadVariableOp?Adam/lstm_7/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOp3Adam/lstm_7/lstm_cell_13/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOp5Adam/lstm_6/lstm_cell_12/kernel/v/Read/ReadVariableOp?Adam/lstm_6/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_6/lstm_cell_12/bias/v/Read/ReadVariableOp5Adam/lstm_7/lstm_cell_13/kernel/v/Read/ReadVariableOp?Adam/lstm_7/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOp3Adam/lstm_7/lstm_cell_13/bias/v/Read/ReadVariableOpConst*N
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
__inference__traced_save_73536
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biasdense_11/kerneldense_11/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_6/lstm_cell_12/kernel$lstm_6/lstm_cell_12/recurrent_kernellstm_6/lstm_cell_12/biaslstm_7/lstm_cell_13/kernel$lstm_7/lstm_cell_13/recurrent_kernellstm_7/lstm_cell_13/biastotalcounttotal_1count_1total_2count_2Adam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/dense_9/kernel/mAdam/dense_9/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/dense_11/kernel/mAdam/dense_11/bias/m!Adam/lstm_6/lstm_cell_12/kernel/m+Adam/lstm_6/lstm_cell_12/recurrent_kernel/mAdam/lstm_6/lstm_cell_12/bias/m!Adam/lstm_7/lstm_cell_13/kernel/m+Adam/lstm_7/lstm_cell_13/recurrent_kernel/mAdam/lstm_7/lstm_cell_13/bias/mAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/dense_9/kernel/vAdam/dense_9/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/vAdam/dense_11/kernel/vAdam/dense_11/bias/v!Adam/lstm_6/lstm_cell_12/kernel/v+Adam/lstm_6/lstm_cell_12/recurrent_kernel/vAdam/lstm_6/lstm_cell_12/bias/v!Adam/lstm_7/lstm_cell_13/kernel/v+Adam/lstm_7/lstm_cell_13/recurrent_kernel/vAdam/lstm_7/lstm_cell_13/bias/v*M
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
!__inference__traced_restore_73741??
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_71684

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
&__inference_lstm_7_layer_call_fn_73202

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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_71516

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
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_71795

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_5_layer_call_fn_71365

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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_713592
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
?K
?
B__inference_model_1_layer_call_and_return_conditional_losses_72188
conv2d_3_input
conv2d_4_input
conv2d_5_input
conv2d_4_72129
conv2d_4_72131
conv2d_3_72134
conv2d_3_72136
conv2d_5_72139
conv2d_5_72141
dense_9_72156
dense_9_72158
lstm_6_72163
lstm_6_72165
lstm_6_72167
dense_10_72170
dense_10_72172
lstm_7_72175
lstm_7_72177
lstm_7_72179
dense_11_72182
dense_11_72184
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?lstm_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_72129conv2d_4_72131*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_717092"
 conv2d_4/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_72134conv2d_3_72136*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_717352"
 conv2d_3/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_72139conv2d_5_72141*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_717612"
 conv2d_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_717822
activation_4/PartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_717952
activation_3/PartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_718082
activation_5/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_713472!
max_pooling2d_4/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_713352!
max_pooling2d_3/PartitionedCally
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul(max_pooling2d_3/PartitionedCall:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul(max_pooling2d_4/PartitionedCall:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMul%activation_5/PartitionedCall:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mul?
concatenate_1/PartitionedCallPartitionedCalltf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_718322
concatenate_1/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9_72156dense_9_72158*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_718532!
dense_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_713592!
max_pooling2d_5/PartitionedCall?
tf.compat.v1.squeeze_1/SqueezeSqueeze(max_pooling2d_5/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeeze?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_1/Squeeze:output:0lstm_6_72163lstm_6_72165lstm_6_72167*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719382 
lstm_6/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_10_72170dense_10_72172*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_719792"
 dense_10/StatefulPartitionedCall?
lstm_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0lstm_7_72175lstm_7_72177lstm_7_72179*
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720662 
lstm_7/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall'lstm_7/StatefulPartitionedCall:output:0dense_11_72182dense_11_72184*
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
GPU 2J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_721072"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall^lstm_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2@
lstm_7/StatefulPartitionedCalllstm_7/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
?!
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_73032
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
&__inference_lstm_7_layer_call_fn_73191

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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs

?
__inference__traced_save_73536
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop9
5savev2_lstm_6_lstm_cell_12_kernel_read_readvariableopC
?savev2_lstm_6_lstm_cell_12_recurrent_kernel_read_readvariableop7
3savev2_lstm_6_lstm_cell_12_bias_read_readvariableop9
5savev2_lstm_7_lstm_cell_13_kernel_read_readvariableopC
?savev2_lstm_7_lstm_cell_13_recurrent_kernel_read_readvariableop7
3savev2_lstm_7_lstm_cell_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop5
1savev2_adam_dense_11_kernel_m_read_readvariableop3
/savev2_adam_dense_11_bias_m_read_readvariableop@
<savev2_adam_lstm_6_lstm_cell_12_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_6_lstm_cell_12_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_6_lstm_cell_12_bias_m_read_readvariableop@
<savev2_adam_lstm_7_lstm_cell_13_kernel_m_read_readvariableopJ
Fsavev2_adam_lstm_7_lstm_cell_13_recurrent_kernel_m_read_readvariableop>
:savev2_adam_lstm_7_lstm_cell_13_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop5
1savev2_adam_dense_11_kernel_v_read_readvariableop3
/savev2_adam_dense_11_bias_v_read_readvariableop@
<savev2_adam_lstm_6_lstm_cell_12_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_6_lstm_cell_12_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_6_lstm_cell_12_bias_v_read_readvariableop@
<savev2_adam_lstm_7_lstm_cell_13_kernel_v_read_readvariableopJ
Fsavev2_adam_lstm_7_lstm_cell_13_recurrent_kernel_v_read_readvariableop>
:savev2_adam_lstm_7_lstm_cell_13_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop5savev2_lstm_6_lstm_cell_12_kernel_read_readvariableop?savev2_lstm_6_lstm_cell_12_recurrent_kernel_read_readvariableop3savev2_lstm_6_lstm_cell_12_bias_read_readvariableop5savev2_lstm_7_lstm_cell_13_kernel_read_readvariableop?savev2_lstm_7_lstm_cell_13_recurrent_kernel_read_readvariableop3savev2_lstm_7_lstm_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop1savev2_adam_dense_11_kernel_m_read_readvariableop/savev2_adam_dense_11_bias_m_read_readvariableop<savev2_adam_lstm_6_lstm_cell_12_kernel_m_read_readvariableopFsavev2_adam_lstm_6_lstm_cell_12_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_6_lstm_cell_12_bias_m_read_readvariableop<savev2_adam_lstm_7_lstm_cell_13_kernel_m_read_readvariableopFsavev2_adam_lstm_7_lstm_cell_13_recurrent_kernel_m_read_readvariableop:savev2_adam_lstm_7_lstm_cell_13_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop1savev2_adam_dense_11_kernel_v_read_readvariableop/savev2_adam_dense_11_bias_v_read_readvariableop<savev2_adam_lstm_6_lstm_cell_12_kernel_v_read_readvariableopFsavev2_adam_lstm_6_lstm_cell_12_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_6_lstm_cell_12_bias_v_read_readvariableop<savev2_adam_lstm_7_lstm_cell_13_kernel_v_read_readvariableopFsavev2_adam_lstm_7_lstm_cell_13_recurrent_kernel_v_read_readvariableop:savev2_adam_lstm_7_lstm_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?K
?
B__inference_model_1_layer_call_and_return_conditional_losses_72364

inputs
inputs_1
inputs_2
conv2d_4_72305
conv2d_4_72307
conv2d_3_72310
conv2d_3_72312
conv2d_5_72315
conv2d_5_72317
dense_9_72332
dense_9_72334
lstm_6_72339
lstm_6_72341
lstm_6_72343
dense_10_72346
dense_10_72348
lstm_7_72351
lstm_7_72353
lstm_7_72355
dense_11_72358
dense_11_72360
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?lstm_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_4_72305conv2d_4_72307*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_717092"
 conv2d_4/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_72310conv2d_3_72312*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_717352"
 conv2d_3/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_5_72315conv2d_5_72317*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_717612"
 conv2d_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_717822
activation_4/PartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_717952
activation_3/PartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_718082
activation_5/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_713472!
max_pooling2d_4/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_713352!
max_pooling2d_3/PartitionedCally
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul(max_pooling2d_3/PartitionedCall:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul(max_pooling2d_4/PartitionedCall:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMul%activation_5/PartitionedCall:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mul?
concatenate_1/PartitionedCallPartitionedCalltf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_718322
concatenate_1/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9_72332dense_9_72334*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_718532!
dense_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_713592!
max_pooling2d_5/PartitionedCall?
tf.compat.v1.squeeze_1/SqueezeSqueeze(max_pooling2d_5/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeeze?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_1/Squeeze:output:0lstm_6_72339lstm_6_72341lstm_6_72343*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719382 
lstm_6/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_10_72346dense_10_72348*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_719792"
 dense_10/StatefulPartitionedCall?
lstm_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0lstm_7_72351lstm_7_72353lstm_7_72355*
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720662 
lstm_7/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall'lstm_7/StatefulPartitionedCall:output:0dense_11_72358dense_11_72360*
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
GPU 2J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_721072"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall^lstm_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2@
lstm_7/StatefulPartitionedCalllstm_7/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_6_layer_call_fn_73077
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
A__inference_lstm_6_layer_call_and_return_conditional_losses_714712
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
}
(__inference_conv2d_4_layer_call_fn_72824

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
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_717092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73180

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
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_71709

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
-__inference_concatenate_1_layer_call_fn_72888
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
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_718322
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/2
?
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_71359

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
C__inference_dense_11_layer_call_and_return_conditional_losses_73307

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
H
,__inference_activation_5_layer_call_fn_72873

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
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_718082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?#
!__inference__traced_restore_73741
file_prefix$
 assignvariableop_conv2d_3_kernel$
 assignvariableop_1_conv2d_3_bias&
"assignvariableop_2_conv2d_4_kernel$
 assignvariableop_3_conv2d_4_bias&
"assignvariableop_4_conv2d_5_kernel$
 assignvariableop_5_conv2d_5_bias%
!assignvariableop_6_dense_9_kernel#
assignvariableop_7_dense_9_bias&
"assignvariableop_8_dense_10_kernel$
 assignvariableop_9_dense_10_bias'
#assignvariableop_10_dense_11_kernel%
!assignvariableop_11_dense_11_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate2
.assignvariableop_17_lstm_6_lstm_cell_12_kernel<
8assignvariableop_18_lstm_6_lstm_cell_12_recurrent_kernel0
,assignvariableop_19_lstm_6_lstm_cell_12_bias2
.assignvariableop_20_lstm_7_lstm_cell_13_kernel<
8assignvariableop_21_lstm_7_lstm_cell_13_recurrent_kernel0
,assignvariableop_22_lstm_7_lstm_cell_13_bias
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1
assignvariableop_27_total_2
assignvariableop_28_count_2.
*assignvariableop_29_adam_conv2d_3_kernel_m,
(assignvariableop_30_adam_conv2d_3_bias_m.
*assignvariableop_31_adam_conv2d_4_kernel_m,
(assignvariableop_32_adam_conv2d_4_bias_m.
*assignvariableop_33_adam_conv2d_5_kernel_m,
(assignvariableop_34_adam_conv2d_5_bias_m-
)assignvariableop_35_adam_dense_9_kernel_m+
'assignvariableop_36_adam_dense_9_bias_m.
*assignvariableop_37_adam_dense_10_kernel_m,
(assignvariableop_38_adam_dense_10_bias_m.
*assignvariableop_39_adam_dense_11_kernel_m,
(assignvariableop_40_adam_dense_11_bias_m9
5assignvariableop_41_adam_lstm_6_lstm_cell_12_kernel_mC
?assignvariableop_42_adam_lstm_6_lstm_cell_12_recurrent_kernel_m7
3assignvariableop_43_adam_lstm_6_lstm_cell_12_bias_m9
5assignvariableop_44_adam_lstm_7_lstm_cell_13_kernel_mC
?assignvariableop_45_adam_lstm_7_lstm_cell_13_recurrent_kernel_m7
3assignvariableop_46_adam_lstm_7_lstm_cell_13_bias_m.
*assignvariableop_47_adam_conv2d_3_kernel_v,
(assignvariableop_48_adam_conv2d_3_bias_v.
*assignvariableop_49_adam_conv2d_4_kernel_v,
(assignvariableop_50_adam_conv2d_4_bias_v.
*assignvariableop_51_adam_conv2d_5_kernel_v,
(assignvariableop_52_adam_conv2d_5_bias_v-
)assignvariableop_53_adam_dense_9_kernel_v+
'assignvariableop_54_adam_dense_9_bias_v.
*assignvariableop_55_adam_dense_10_kernel_v,
(assignvariableop_56_adam_dense_10_bias_v.
*assignvariableop_57_adam_dense_11_kernel_v,
(assignvariableop_58_adam_dense_11_bias_v9
5assignvariableop_59_adam_lstm_6_lstm_cell_12_kernel_vC
?assignvariableop_60_adam_lstm_6_lstm_cell_12_recurrent_kernel_v7
3assignvariableop_61_adam_lstm_6_lstm_cell_12_bias_v9
5assignvariableop_62_adam_lstm_7_lstm_cell_13_kernel_vC
?assignvariableop_63_adam_lstm_7_lstm_cell_13_recurrent_kernel_v7
3assignvariableop_64_adam_lstm_7_lstm_cell_13_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_10_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_10_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_11_biasIdentity_11:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp.assignvariableop_17_lstm_6_lstm_cell_12_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp8assignvariableop_18_lstm_6_lstm_cell_12_recurrent_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_lstm_6_lstm_cell_12_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp.assignvariableop_20_lstm_7_lstm_cell_13_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp8assignvariableop_21_lstm_7_lstm_cell_13_recurrent_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_lstm_7_lstm_cell_13_biasIdentity_22:output:0"/device:CPU:0*
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
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_3_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_5_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_5_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_9_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_9_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_10_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_10_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_11_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_11_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adam_lstm_6_lstm_cell_12_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp?assignvariableop_42_adam_lstm_6_lstm_cell_12_recurrent_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp3assignvariableop_43_adam_lstm_6_lstm_cell_12_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_lstm_7_lstm_cell_13_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_lstm_7_lstm_cell_13_recurrent_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp3assignvariableop_46_adam_lstm_7_lstm_cell_13_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv2d_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv2d_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_5_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_5_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_9_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_9_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_10_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_10_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_11_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_11_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_lstm_6_lstm_cell_12_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp?assignvariableop_60_adam_lstm_6_lstm_cell_12_recurrent_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp3assignvariableop_61_adam_lstm_6_lstm_cell_12_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_lstm_7_lstm_cell_13_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp?assignvariableop_63_adam_lstm_7_lstm_cell_13_recurrent_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp3assignvariableop_64_adam_lstm_7_lstm_cell_13_bias_vIdentity_64:output:0"/device:CPU:0*
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
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_71832

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
:??????????`2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_72786
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
GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_723642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
c
G__inference_activation_3_layer_call_and_return_conditional_losses_72829

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_72066

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
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
|
'__inference_dense_9_layer_call_fn_72908

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
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_718532
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_72578
inputs_0
inputs_1
inputs_2+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource*
&lstm_6_mlclstm_readvariableop_resource,
(lstm_6_mlclstm_readvariableop_1_resource,
(lstm_6_mlclstm_readvariableop_2_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource*
&lstm_7_mlclstm_readvariableop_resource,
(lstm_7_mlclstm_readvariableop_1_resource,
(lstm_7_mlclstm_readvariableop_2_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?lstm_6/MLCLSTM/ReadVariableOp?lstm_6/MLCLSTM/ReadVariableOp_1?lstm_6/MLCLSTM/ReadVariableOp_2?lstm_7/MLCLSTM/ReadVariableOp?lstm_7/MLCLSTM/ReadVariableOp_1?lstm_7/MLCLSTM/ReadVariableOp_2?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2D	MLCConv2Dinputs_1&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_4/BiasAdd?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D	MLCConv2Dinputs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_3/BiasAdd?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D	MLCConv2Dinputs_2&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_5/BiasAdd?
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_4/Relu?
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_3/Relu?
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_5/Relu?
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPooly
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul max_pooling2d_3/MaxPool:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul max_pooling2d_4/MaxPool:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMulactivation_5/Relu:activations:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mulx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2tf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate_1/concat?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMulconcatenate_1/concat:output:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_9/BiasAddy
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_9/Tanh?
max_pooling2d_5/MaxPoolMaxPooldense_9/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
tf.compat.v1.squeeze_1/SqueezeSqueeze max_pooling2d_5/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeezes
lstm_6/ShapeShape'tf.compat.v1.squeeze_1/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_6/Shape?
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack?
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1?
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2?
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicek
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/mul/y?
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/Less/y?
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessq
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/packed/1?
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const?
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_6/zeroso
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/mul/y?
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/Less/y?
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lessu
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/packed/1?
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const?
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_6/zeros_1{
lstm_6/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/MLCLSTM/hidden_size{
lstm_6/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/MLCLSTM/output_size?
lstm_6/MLCLSTM/ReadVariableOpReadVariableOp&lstm_6_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
lstm_6/MLCLSTM/ReadVariableOp?
lstm_6/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_6_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_6/MLCLSTM/ReadVariableOp_1?
lstm_6/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_6_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_6/MLCLSTM/ReadVariableOp_2?
lstm_6/MLCLSTMMLCLSTM'tf.compat.v1.squeeze_1/Squeeze:output:0#lstm_6/MLCLSTM/hidden_size:output:0#lstm_6/MLCLSTM/output_size:output:0%lstm_6/MLCLSTM/ReadVariableOp:value:0'lstm_6/MLCLSTM/ReadVariableOp_1:value:0'lstm_6/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_6/MLCLSTM?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMullstm_6/MLCLSTM:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_10/BiasAddx
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_10/Tanh]
lstm_7/ShapeShapedense_10/Tanh:y:0*
T0*
_output_shapes
:2
lstm_7/Shape?
lstm_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_7/strided_slice/stack?
lstm_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_7/strided_slice/stack_1?
lstm_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_7/strided_slice/stack_2?
lstm_7/strided_sliceStridedSlicelstm_7/Shape:output:0#lstm_7/strided_slice/stack:output:0%lstm_7/strided_slice/stack_1:output:0%lstm_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_7/strided_slicej
lstm_7/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros/mul/y?
lstm_7/zeros/mulMullstm_7/strided_slice:output:0lstm_7/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros/mulm
lstm_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_7/zeros/Less/y?
lstm_7/zeros/LessLesslstm_7/zeros/mul:z:0lstm_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros/Lessp
lstm_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros/packed/1?
lstm_7/zeros/packedPacklstm_7/strided_slice:output:0lstm_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_7/zeros/packedm
lstm_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_7/zeros/Const?
lstm_7/zerosFilllstm_7/zeros/packed:output:0lstm_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/zerosn
lstm_7/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros_1/mul/y?
lstm_7/zeros_1/mulMullstm_7/strided_slice:output:0lstm_7/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros_1/mulq
lstm_7/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_7/zeros_1/Less/y?
lstm_7/zeros_1/LessLesslstm_7/zeros_1/mul:z:0lstm_7/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros_1/Lesst
lstm_7/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros_1/packed/1?
lstm_7/zeros_1/packedPacklstm_7/strided_slice:output:0 lstm_7/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_7/zeros_1/packedq
lstm_7/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_7/zeros_1/Const?
lstm_7/zeros_1Filllstm_7/zeros_1/packed:output:0lstm_7/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/zeros_1z
lstm_7/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/MLCLSTM/hidden_sizez
lstm_7/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/MLCLSTM/output_size?
lstm_7/MLCLSTM/ReadVariableOpReadVariableOp&lstm_7_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_7/MLCLSTM/ReadVariableOp?
lstm_7/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_7_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02!
lstm_7/MLCLSTM/ReadVariableOp_1?
lstm_7/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_7_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_7/MLCLSTM/ReadVariableOp_2?
lstm_7/MLCLSTMMLCLSTMdense_10/Tanh:y:0#lstm_7/MLCLSTM/hidden_size:output:0#lstm_7/MLCLSTM/output_size:output:0%lstm_7/MLCLSTM/ReadVariableOp:value:0'lstm_7/MLCLSTM/ReadVariableOp_1:value:0'lstm_7/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_7/MLCLSTM}
lstm_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_7/Reshape/shape?
lstm_7/ReshapeReshapelstm_7/MLCLSTM:output:0lstm_7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/Reshape?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMullstm_7/Reshape:output:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/BiasAddt
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_11/Tanh?
IdentityIdentitydense_11/Tanh:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp^lstm_6/MLCLSTM/ReadVariableOp ^lstm_6/MLCLSTM/ReadVariableOp_1 ^lstm_6/MLCLSTM/ReadVariableOp_2^lstm_7/MLCLSTM/ReadVariableOp ^lstm_7/MLCLSTM/ReadVariableOp_1 ^lstm_7/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp2>
lstm_6/MLCLSTM/ReadVariableOplstm_6/MLCLSTM/ReadVariableOp2B
lstm_6/MLCLSTM/ReadVariableOp_1lstm_6/MLCLSTM/ReadVariableOp_12B
lstm_6/MLCLSTM/ReadVariableOp_2lstm_6/MLCLSTM/ReadVariableOp_22>
lstm_7/MLCLSTM/ReadVariableOplstm_7/MLCLSTM/ReadVariableOp2B
lstm_7/MLCLSTM/ReadVariableOp_1lstm_7/MLCLSTM/ReadVariableOp_12B
lstm_7/MLCLSTM/ReadVariableOp_2lstm_7/MLCLSTM/ReadVariableOp_2:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?
}
(__inference_dense_10_layer_call_fn_73108

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
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_719792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73274
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
B__inference_dense_9_layer_call_and_return_conditional_losses_72899

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
:??????????0*

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
:??????????02	
BiasAdda
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
?

?
C__inference_dense_10_layer_call_and_return_conditional_losses_71979

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
:??????????*

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
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_72839

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73144

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
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_71471

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
&__inference_lstm_6_layer_call_fn_73088
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
A__inference_lstm_6_layer_call_and_return_conditional_losses_715162
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
?

?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_72854

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_72403
conv2d_3_input
conv2d_4_input
conv2d_5_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_4_inputconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_723642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
? 
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_72942

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
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_72868

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
H
,__inference_activation_3_layer_call_fn_72834

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
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_717952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
}
(__inference_conv2d_5_layer_call_fn_72863

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
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_717612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_72296
conv2d_3_input
conv2d_4_input
conv2d_5_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_4_inputconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_722572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
?
}
(__inference_dense_11_layer_call_fn_73316

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
GPU 2J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_721072
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
?
?
&__inference_lstm_7_layer_call_fn_73296
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_716842
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
?K
?
B__inference_model_1_layer_call_and_return_conditional_losses_72124
conv2d_3_input
conv2d_4_input
conv2d_5_input
conv2d_4_71720
conv2d_4_71722
conv2d_3_71746
conv2d_3_71748
conv2d_5_71772
conv2d_5_71774
dense_9_71864
dense_9_71866
lstm_6_71961
lstm_6_71963
lstm_6_71965
dense_10_71990
dense_10_71992
lstm_7_72089
lstm_7_72091
lstm_7_72093
dense_11_72118
dense_11_72120
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?lstm_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_71720conv2d_4_71722*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_717092"
 conv2d_4/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_71746conv2d_3_71748*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_717352"
 conv2d_3/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_71772conv2d_5_71774*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_717612"
 conv2d_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_717822
activation_4/PartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_717952
activation_3/PartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_718082
activation_5/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_713472!
max_pooling2d_4/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_713352!
max_pooling2d_3/PartitionedCally
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul(max_pooling2d_3/PartitionedCall:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul(max_pooling2d_4/PartitionedCall:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMul%activation_5/PartitionedCall:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mul?
concatenate_1/PartitionedCallPartitionedCalltf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_718322
concatenate_1/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9_71864dense_9_71866*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_718532!
dense_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_713592!
max_pooling2d_5/PartitionedCall?
tf.compat.v1.squeeze_1/SqueezeSqueeze(max_pooling2d_5/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeeze?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_1/Squeeze:output:0lstm_6_71961lstm_6_71963lstm_6_71965*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719042 
lstm_6/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_10_71990dense_10_71992*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_719792"
 dense_10/StatefulPartitionedCall?
lstm_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0lstm_7_72089lstm_7_72091lstm_7_72093*
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720302 
lstm_7/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall'lstm_7/StatefulPartitionedCall:output:0dense_11_72118dense_11_72120*
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
GPU 2J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_721072"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall^lstm_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2@
lstm_7/StatefulPartitionedCalllstm_7/StatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
?
c
G__inference_activation_4_layer_call_and_return_conditional_losses_71782

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_72815

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_7_layer_call_fn_73285
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_716372
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
B__inference_dense_9_layer_call_and_return_conditional_losses_71853

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
:??????????0*

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
:??????????02	
BiasAdda
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:??????????02
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*0
_output_shapes
:??????????02

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????`::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????`
 
_user_specified_nameinputs
?
c
G__inference_activation_5_layer_call_and_return_conditional_losses_71808

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????? 2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
? 
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_71938

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
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?

?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_71761

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?K
?
B__inference_model_1_layer_call_and_return_conditional_losses_72257

inputs
inputs_1
inputs_2
conv2d_4_72198
conv2d_4_72200
conv2d_3_72203
conv2d_3_72205
conv2d_5_72208
conv2d_5_72210
dense_9_72225
dense_9_72227
lstm_6_72232
lstm_6_72234
lstm_6_72236
dense_10_72239
dense_10_72241
lstm_7_72244
lstm_7_72246
lstm_7_72248
dense_11_72251
dense_11_72253
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? dense_10/StatefulPartitionedCall? dense_11/StatefulPartitionedCall?dense_9/StatefulPartitionedCall?lstm_6/StatefulPartitionedCall?lstm_7/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_4_72198conv2d_4_72200*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_4_layer_call_and_return_conditional_losses_717092"
 conv2d_4/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_72203conv2d_3_72205*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_717352"
 conv2d_3/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_5_72208conv2d_5_72210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_717612"
 conv2d_5/StatefulPartitionedCall?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_717822
activation_4/PartitionedCall?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_717952
activation_3/PartitionedCall?
activation_5/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_718082
activation_5/PartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_713472!
max_pooling2d_4/PartitionedCall?
max_pooling2d_3/PartitionedCallPartitionedCall%activation_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_713352!
max_pooling2d_3/PartitionedCally
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul(max_pooling2d_3/PartitionedCall:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul(max_pooling2d_4/PartitionedCall:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMul%activation_5/PartitionedCall:output:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mul?
concatenate_1/PartitionedCallPartitionedCalltf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_718322
concatenate_1/PartitionedCall?
dense_9/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9_72225dense_9_72227*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_718532!
dense_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_713592!
max_pooling2d_5/PartitionedCall?
tf.compat.v1.squeeze_1/SqueezeSqueeze(max_pooling2d_5/PartitionedCall:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeeze?
lstm_6/StatefulPartitionedCallStatefulPartitionedCall'tf.compat.v1.squeeze_1/Squeeze:output:0lstm_6_72232lstm_6_72234lstm_6_72236*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719042 
lstm_6/StatefulPartitionedCall?
 dense_10/StatefulPartitionedCallStatefulPartitionedCall'lstm_6/StatefulPartitionedCall:output:0dense_10_72239dense_10_72241*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_719792"
 dense_10/StatefulPartitionedCall?
lstm_7/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0lstm_7_72244lstm_7_72246lstm_7_72248*
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_720302 
lstm_7/StatefulPartitionedCall?
 dense_11/StatefulPartitionedCallStatefulPartitionedCall'lstm_7/StatefulPartitionedCall:output:0dense_11_72251dense_11_72253*
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
GPU 2J 8? *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_721072"
 dense_11/StatefulPartitionedCall?
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm_6/StatefulPartitionedCall^lstm_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2@
lstm_6/StatefulPartitionedCalllstm_6/StatefulPartitionedCall2@
lstm_7/StatefulPartitionedCalllstm_7/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_72743
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
GPU 2J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_722572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_72030

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
$:??????????:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_71347

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
C__inference_dense_11_layer_call_and_return_conditional_losses_72107

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
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_72881
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
:??????????`2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????`2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/2
?
K
/__inference_max_pooling2d_3_layer_call_fn_71341

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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_713352
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
?!
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_73066
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
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_71735

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_lstm_6_layer_call_fn_72987

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
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?

?
C__inference_dense_10_layer_call_and_return_conditional_losses_73099

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
:??????????*

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
:??????????2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:??????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_72976

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
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73238
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
?"
?
A__inference_lstm_7_layer_call_and_return_conditional_losses_71637

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
??
?
 __inference__wrapped_model_71329
conv2d_3_input
conv2d_4_input
conv2d_5_input3
/model_1_conv2d_4_conv2d_readvariableop_resource4
0model_1_conv2d_4_biasadd_readvariableop_resource3
/model_1_conv2d_3_conv2d_readvariableop_resource4
0model_1_conv2d_3_biasadd_readvariableop_resource3
/model_1_conv2d_5_conv2d_readvariableop_resource4
0model_1_conv2d_5_biasadd_readvariableop_resource5
1model_1_dense_9_mlcmatmul_readvariableop_resource3
/model_1_dense_9_biasadd_readvariableop_resource2
.model_1_lstm_6_mlclstm_readvariableop_resource4
0model_1_lstm_6_mlclstm_readvariableop_1_resource4
0model_1_lstm_6_mlclstm_readvariableop_2_resource6
2model_1_dense_10_mlcmatmul_readvariableop_resource4
0model_1_dense_10_biasadd_readvariableop_resource2
.model_1_lstm_7_mlclstm_readvariableop_resource4
0model_1_lstm_7_mlclstm_readvariableop_1_resource4
0model_1_lstm_7_mlclstm_readvariableop_2_resource6
2model_1_dense_11_mlcmatmul_readvariableop_resource4
0model_1_dense_11_biasadd_readvariableop_resource
identity??'model_1/conv2d_3/BiasAdd/ReadVariableOp?&model_1/conv2d_3/Conv2D/ReadVariableOp?'model_1/conv2d_4/BiasAdd/ReadVariableOp?&model_1/conv2d_4/Conv2D/ReadVariableOp?'model_1/conv2d_5/BiasAdd/ReadVariableOp?&model_1/conv2d_5/Conv2D/ReadVariableOp?'model_1/dense_10/BiasAdd/ReadVariableOp?)model_1/dense_10/MLCMatMul/ReadVariableOp?'model_1/dense_11/BiasAdd/ReadVariableOp?)model_1/dense_11/MLCMatMul/ReadVariableOp?&model_1/dense_9/BiasAdd/ReadVariableOp?(model_1/dense_9/MLCMatMul/ReadVariableOp?%model_1/lstm_6/MLCLSTM/ReadVariableOp?'model_1/lstm_6/MLCLSTM/ReadVariableOp_1?'model_1/lstm_6/MLCLSTM/ReadVariableOp_2?%model_1/lstm_7/MLCLSTM/ReadVariableOp?'model_1/lstm_7/MLCLSTM/ReadVariableOp_1?'model_1/lstm_7/MLCLSTM/ReadVariableOp_2?
&model_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_1/conv2d_4/Conv2D/ReadVariableOp?
model_1/conv2d_4/Conv2D	MLCConv2Dconv2d_4_input.model_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_1/conv2d_4/Conv2D?
'model_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_4/BiasAdd/ReadVariableOp?
model_1/conv2d_4/BiasAddBiasAdd model_1/conv2d_4/Conv2D:output:0/model_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_1/conv2d_4/BiasAdd?
&model_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_1/conv2d_3/Conv2D/ReadVariableOp?
model_1/conv2d_3/Conv2D	MLCConv2Dconv2d_3_input.model_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_1/conv2d_3/Conv2D?
'model_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_3/BiasAdd/ReadVariableOp?
model_1/conv2d_3/BiasAddBiasAdd model_1/conv2d_3/Conv2D:output:0/model_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_1/conv2d_3/BiasAdd?
&model_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&model_1/conv2d_5/Conv2D/ReadVariableOp?
model_1/conv2d_5/Conv2D	MLCConv2Dconv2d_5_input.model_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
model_1/conv2d_5/Conv2D?
'model_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_1/conv2d_5/BiasAdd/ReadVariableOp?
model_1/conv2d_5/BiasAddBiasAdd model_1/conv2d_5/Conv2D:output:0/model_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
model_1/conv2d_5/BiasAdd?
model_1/activation_4/ReluRelu!model_1/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_1/activation_4/Relu?
model_1/activation_3/ReluRelu!model_1/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_1/activation_3/Relu?
model_1/activation_5/ReluRelu!model_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
model_1/activation_5/Relu?
model_1/max_pooling2d_4/MaxPoolMaxPool'model_1/activation_4/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_4/MaxPool?
model_1/max_pooling2d_3/MaxPoolMaxPool'model_1/activation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_3/MaxPool?
 model_1/tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model_1/tf.math.multiply_3/Mul/y?
model_1/tf.math.multiply_3/MulMul(model_1/max_pooling2d_3/MaxPool:output:0)model_1/tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2 
model_1/tf.math.multiply_3/Mul?
 model_1/tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2"
 model_1/tf.math.multiply_4/Mul/y?
model_1/tf.math.multiply_4/MulMul(model_1/max_pooling2d_4/MaxPool:output:0)model_1/tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2 
model_1/tf.math.multiply_4/Mul?
 model_1/tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2"
 model_1/tf.math.multiply_5/Mul/y?
model_1/tf.math.multiply_5/MulMul'model_1/activation_5/Relu:activations:0)model_1/tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2 
model_1/tf.math.multiply_5/Mul?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2"model_1/tf.math.multiply_3/Mul:z:0"model_1/tf.math.multiply_4/Mul:z:0"model_1/tf.math.multiply_5/Mul:z:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
model_1/concatenate_1/concat?
(model_1/dense_9/MLCMatMul/ReadVariableOpReadVariableOp1model_1_dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02*
(model_1/dense_9/MLCMatMul/ReadVariableOp?
model_1/dense_9/MLCMatMul	MLCMatMul%model_1/concatenate_1/concat:output:00model_1/dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
model_1/dense_9/MLCMatMul?
&model_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02(
&model_1/dense_9/BiasAdd/ReadVariableOp?
model_1/dense_9/BiasAddBiasAdd#model_1/dense_9/MLCMatMul:product:0.model_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
model_1/dense_9/BiasAdd?
model_1/dense_9/TanhTanh model_1/dense_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
model_1/dense_9/Tanh?
model_1/max_pooling2d_5/MaxPoolMaxPoolmodel_1/dense_9/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_5/MaxPool?
&model_1/tf.compat.v1.squeeze_1/SqueezeSqueeze(model_1/max_pooling2d_5/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2(
&model_1/tf.compat.v1.squeeze_1/Squeeze?
model_1/lstm_6/ShapeShape/model_1/tf.compat.v1.squeeze_1/Squeeze:output:0*
T0*
_output_shapes
:2
model_1/lstm_6/Shape?
"model_1/lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_1/lstm_6/strided_slice/stack?
$model_1/lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_1/lstm_6/strided_slice/stack_1?
$model_1/lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_1/lstm_6/strided_slice/stack_2?
model_1/lstm_6/strided_sliceStridedSlicemodel_1/lstm_6/Shape:output:0+model_1/lstm_6/strided_slice/stack:output:0-model_1/lstm_6/strided_slice/stack_1:output:0-model_1/lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_1/lstm_6/strided_slice{
model_1/lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_6/zeros/mul/y?
model_1/lstm_6/zeros/mulMul%model_1/lstm_6/strided_slice:output:0#model_1/lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_6/zeros/mul}
model_1/lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_6/zeros/Less/y?
model_1/lstm_6/zeros/LessLessmodel_1/lstm_6/zeros/mul:z:0$model_1/lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_6/zeros/Less?
model_1/lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_6/zeros/packed/1?
model_1/lstm_6/zeros/packedPack%model_1/lstm_6/strided_slice:output:0&model_1/lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_1/lstm_6/zeros/packed}
model_1/lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_6/zeros/Const?
model_1/lstm_6/zerosFill$model_1/lstm_6/zeros/packed:output:0#model_1/lstm_6/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/lstm_6/zeros
model_1/lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_6/zeros_1/mul/y?
model_1/lstm_6/zeros_1/mulMul%model_1/lstm_6/strided_slice:output:0%model_1/lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_6/zeros_1/mul?
model_1/lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_6/zeros_1/Less/y?
model_1/lstm_6/zeros_1/LessLessmodel_1/lstm_6/zeros_1/mul:z:0&model_1/lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_6/zeros_1/Less?
model_1/lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
model_1/lstm_6/zeros_1/packed/1?
model_1/lstm_6/zeros_1/packedPack%model_1/lstm_6/strided_slice:output:0(model_1/lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_1/lstm_6/zeros_1/packed?
model_1/lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_6/zeros_1/Const?
model_1/lstm_6/zeros_1Fill&model_1/lstm_6/zeros_1/packed:output:0%model_1/lstm_6/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/lstm_6/zeros_1?
"model_1/lstm_6/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/lstm_6/MLCLSTM/hidden_size?
"model_1/lstm_6/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2$
"model_1/lstm_6/MLCLSTM/output_size?
%model_1/lstm_6/MLCLSTM/ReadVariableOpReadVariableOp.model_1_lstm_6_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02'
%model_1/lstm_6/MLCLSTM/ReadVariableOp?
'model_1/lstm_6/MLCLSTM/ReadVariableOp_1ReadVariableOp0model_1_lstm_6_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02)
'model_1/lstm_6/MLCLSTM/ReadVariableOp_1?
'model_1/lstm_6/MLCLSTM/ReadVariableOp_2ReadVariableOp0model_1_lstm_6_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02)
'model_1/lstm_6/MLCLSTM/ReadVariableOp_2?
model_1/lstm_6/MLCLSTMMLCLSTM/model_1/tf.compat.v1.squeeze_1/Squeeze:output:0+model_1/lstm_6/MLCLSTM/hidden_size:output:0+model_1/lstm_6/MLCLSTM/output_size:output:0-model_1/lstm_6/MLCLSTM/ReadVariableOp:value:0/model_1/lstm_6/MLCLSTM/ReadVariableOp_1:value:0/model_1/lstm_6/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
model_1/lstm_6/MLCLSTM?
)model_1/dense_10/MLCMatMul/ReadVariableOpReadVariableOp2model_1_dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02+
)model_1/dense_10/MLCMatMul/ReadVariableOp?
model_1/dense_10/MLCMatMul	MLCMatMulmodel_1/lstm_6/MLCLSTM:output:01model_1/dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
model_1/dense_10/MLCMatMul?
'model_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/dense_10/BiasAdd/ReadVariableOp?
model_1/dense_10/BiasAddBiasAdd$model_1/dense_10/MLCMatMul:product:0/model_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
model_1/dense_10/BiasAdd?
model_1/dense_10/TanhTanh!model_1/dense_10/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
model_1/dense_10/Tanhu
model_1/lstm_7/ShapeShapemodel_1/dense_10/Tanh:y:0*
T0*
_output_shapes
:2
model_1/lstm_7/Shape?
"model_1/lstm_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model_1/lstm_7/strided_slice/stack?
$model_1/lstm_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_1/lstm_7/strided_slice/stack_1?
$model_1/lstm_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model_1/lstm_7/strided_slice/stack_2?
model_1/lstm_7/strided_sliceStridedSlicemodel_1/lstm_7/Shape:output:0+model_1/lstm_7/strided_slice/stack:output:0-model_1/lstm_7/strided_slice/stack_1:output:0-model_1/lstm_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_1/lstm_7/strided_slicez
model_1/lstm_7/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model_1/lstm_7/zeros/mul/y?
model_1/lstm_7/zeros/mulMul%model_1/lstm_7/strided_slice:output:0#model_1/lstm_7/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_7/zeros/mul}
model_1/lstm_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_7/zeros/Less/y?
model_1/lstm_7/zeros/LessLessmodel_1/lstm_7/zeros/mul:z:0$model_1/lstm_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_7/zeros/Less?
model_1/lstm_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
model_1/lstm_7/zeros/packed/1?
model_1/lstm_7/zeros/packedPack%model_1/lstm_7/strided_slice:output:0&model_1/lstm_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_1/lstm_7/zeros/packed}
model_1/lstm_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_7/zeros/Const?
model_1/lstm_7/zerosFill$model_1/lstm_7/zeros/packed:output:0#model_1/lstm_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
model_1/lstm_7/zeros~
model_1/lstm_7/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
model_1/lstm_7/zeros_1/mul/y?
model_1/lstm_7/zeros_1/mulMul%model_1/lstm_7/strided_slice:output:0%model_1/lstm_7/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_7/zeros_1/mul?
model_1/lstm_7/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model_1/lstm_7/zeros_1/Less/y?
model_1/lstm_7/zeros_1/LessLessmodel_1/lstm_7/zeros_1/mul:z:0&model_1/lstm_7/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
model_1/lstm_7/zeros_1/Less?
model_1/lstm_7/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2!
model_1/lstm_7/zeros_1/packed/1?
model_1/lstm_7/zeros_1/packedPack%model_1/lstm_7/strided_slice:output:0(model_1/lstm_7/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
model_1/lstm_7/zeros_1/packed?
model_1/lstm_7/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/lstm_7/zeros_1/Const?
model_1/lstm_7/zeros_1Fill&model_1/lstm_7/zeros_1/packed:output:0%model_1/lstm_7/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
model_1/lstm_7/zeros_1?
"model_1/lstm_7/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2$
"model_1/lstm_7/MLCLSTM/hidden_size?
"model_1/lstm_7/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2$
"model_1/lstm_7/MLCLSTM/output_size?
%model_1/lstm_7/MLCLSTM/ReadVariableOpReadVariableOp.model_1_lstm_7_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_1/lstm_7/MLCLSTM/ReadVariableOp?
'model_1/lstm_7/MLCLSTM/ReadVariableOp_1ReadVariableOp0model_1_lstm_7_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02)
'model_1/lstm_7/MLCLSTM/ReadVariableOp_1?
'model_1/lstm_7/MLCLSTM/ReadVariableOp_2ReadVariableOp0model_1_lstm_7_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02)
'model_1/lstm_7/MLCLSTM/ReadVariableOp_2?
model_1/lstm_7/MLCLSTMMLCLSTMmodel_1/dense_10/Tanh:y:0+model_1/lstm_7/MLCLSTM/hidden_size:output:0+model_1/lstm_7/MLCLSTM/output_size:output:0-model_1/lstm_7/MLCLSTM/ReadVariableOp:value:0/model_1/lstm_7/MLCLSTM/ReadVariableOp_1:value:0/model_1/lstm_7/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
model_1/lstm_7/MLCLSTM?
model_1/lstm_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
model_1/lstm_7/Reshape/shape?
model_1/lstm_7/ReshapeReshapemodel_1/lstm_7/MLCLSTM:output:0%model_1/lstm_7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
model_1/lstm_7/Reshape?
)model_1/dense_11/MLCMatMul/ReadVariableOpReadVariableOp2model_1_dense_11_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02+
)model_1/dense_11/MLCMatMul/ReadVariableOp?
model_1/dense_11/MLCMatMul	MLCMatMulmodel_1/lstm_7/Reshape:output:01model_1/dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_11/MLCMatMul?
'model_1/dense_11/BiasAdd/ReadVariableOpReadVariableOp0model_1_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'model_1/dense_11/BiasAdd/ReadVariableOp?
model_1/dense_11/BiasAddBiasAdd$model_1/dense_11/MLCMatMul:product:0/model_1/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_1/dense_11/BiasAdd?
model_1/dense_11/TanhTanh!model_1/dense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_1/dense_11/Tanh?
IdentityIdentitymodel_1/dense_11/Tanh:y:0(^model_1/conv2d_3/BiasAdd/ReadVariableOp'^model_1/conv2d_3/Conv2D/ReadVariableOp(^model_1/conv2d_4/BiasAdd/ReadVariableOp'^model_1/conv2d_4/Conv2D/ReadVariableOp(^model_1/conv2d_5/BiasAdd/ReadVariableOp'^model_1/conv2d_5/Conv2D/ReadVariableOp(^model_1/dense_10/BiasAdd/ReadVariableOp*^model_1/dense_10/MLCMatMul/ReadVariableOp(^model_1/dense_11/BiasAdd/ReadVariableOp*^model_1/dense_11/MLCMatMul/ReadVariableOp'^model_1/dense_9/BiasAdd/ReadVariableOp)^model_1/dense_9/MLCMatMul/ReadVariableOp&^model_1/lstm_6/MLCLSTM/ReadVariableOp(^model_1/lstm_6/MLCLSTM/ReadVariableOp_1(^model_1/lstm_6/MLCLSTM/ReadVariableOp_2&^model_1/lstm_7/MLCLSTM/ReadVariableOp(^model_1/lstm_7/MLCLSTM/ReadVariableOp_1(^model_1/lstm_7/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2R
'model_1/conv2d_3/BiasAdd/ReadVariableOp'model_1/conv2d_3/BiasAdd/ReadVariableOp2P
&model_1/conv2d_3/Conv2D/ReadVariableOp&model_1/conv2d_3/Conv2D/ReadVariableOp2R
'model_1/conv2d_4/BiasAdd/ReadVariableOp'model_1/conv2d_4/BiasAdd/ReadVariableOp2P
&model_1/conv2d_4/Conv2D/ReadVariableOp&model_1/conv2d_4/Conv2D/ReadVariableOp2R
'model_1/conv2d_5/BiasAdd/ReadVariableOp'model_1/conv2d_5/BiasAdd/ReadVariableOp2P
&model_1/conv2d_5/Conv2D/ReadVariableOp&model_1/conv2d_5/Conv2D/ReadVariableOp2R
'model_1/dense_10/BiasAdd/ReadVariableOp'model_1/dense_10/BiasAdd/ReadVariableOp2V
)model_1/dense_10/MLCMatMul/ReadVariableOp)model_1/dense_10/MLCMatMul/ReadVariableOp2R
'model_1/dense_11/BiasAdd/ReadVariableOp'model_1/dense_11/BiasAdd/ReadVariableOp2V
)model_1/dense_11/MLCMatMul/ReadVariableOp)model_1/dense_11/MLCMatMul/ReadVariableOp2P
&model_1/dense_9/BiasAdd/ReadVariableOp&model_1/dense_9/BiasAdd/ReadVariableOp2T
(model_1/dense_9/MLCMatMul/ReadVariableOp(model_1/dense_9/MLCMatMul/ReadVariableOp2N
%model_1/lstm_6/MLCLSTM/ReadVariableOp%model_1/lstm_6/MLCLSTM/ReadVariableOp2R
'model_1/lstm_6/MLCLSTM/ReadVariableOp_1'model_1/lstm_6/MLCLSTM/ReadVariableOp_12R
'model_1/lstm_6/MLCLSTM/ReadVariableOp_2'model_1/lstm_6/MLCLSTM/ReadVariableOp_22N
%model_1/lstm_7/MLCLSTM/ReadVariableOp%model_1/lstm_7/MLCLSTM/ReadVariableOp2R
'model_1/lstm_7/MLCLSTM/ReadVariableOp_1'model_1/lstm_7/MLCLSTM/ReadVariableOp_12R
'model_1/lstm_7/MLCLSTM/ReadVariableOp_2'model_1/lstm_7/MLCLSTM/ReadVariableOp_2:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
?
H
,__inference_activation_4_layer_call_fn_72844

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
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_717822
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
&__inference_lstm_6_layer_call_fn_72998

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
:???????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_lstm_6_layer_call_and_return_conditional_losses_719382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
}
(__inference_conv2d_3_layer_call_fn_72805

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
:?????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_717352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_72796

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
:?????????? *
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
:?????????? 2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_71353

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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_713472
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
#__inference_signature_wrapper_72456
conv2d_3_input
conv2d_4_input
conv2d_5_input
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_4_inputconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 __inference__wrapped_model_713292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_3_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_4_input:`\
0
_output_shapes
:??????????
(
_user_specified_nameconv2d_5_input
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_72700
inputs_0
inputs_1
inputs_2+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource-
)dense_9_mlcmatmul_readvariableop_resource+
'dense_9_biasadd_readvariableop_resource*
&lstm_6_mlclstm_readvariableop_resource,
(lstm_6_mlclstm_readvariableop_1_resource,
(lstm_6_mlclstm_readvariableop_2_resource.
*dense_10_mlcmatmul_readvariableop_resource,
(dense_10_biasadd_readvariableop_resource*
&lstm_7_mlclstm_readvariableop_resource,
(lstm_7_mlclstm_readvariableop_1_resource,
(lstm_7_mlclstm_readvariableop_2_resource.
*dense_11_mlcmatmul_readvariableop_resource,
(dense_11_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?dense_10/BiasAdd/ReadVariableOp?!dense_10/MLCMatMul/ReadVariableOp?dense_11/BiasAdd/ReadVariableOp?!dense_11/MLCMatMul/ReadVariableOp?dense_9/BiasAdd/ReadVariableOp? dense_9/MLCMatMul/ReadVariableOp?lstm_6/MLCLSTM/ReadVariableOp?lstm_6/MLCLSTM/ReadVariableOp_1?lstm_6/MLCLSTM/ReadVariableOp_2?lstm_7/MLCLSTM/ReadVariableOp?lstm_7/MLCLSTM/ReadVariableOp_1?lstm_7/MLCLSTM/ReadVariableOp_2?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2D	MLCConv2Dinputs_1&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_4/BiasAdd?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2D	MLCConv2Dinputs_0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_3/BiasAdd?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2D	MLCConv2Dinputs_2&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
num_args *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_5/BiasAdd?
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_4/Relu?
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_3/Relu?
activation_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
activation_5/Relu?
max_pooling2d_4/MaxPoolMaxPoolactivation_4/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
max_pooling2d_3/MaxPoolMaxPoolactivation_3/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPooly
tf.math.multiply_3/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_3/Mul/y?
tf.math.multiply_3/MulMul max_pooling2d_3/MaxPool:output:0!tf.math.multiply_3/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_3/Muly
tf.math.multiply_4/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
tf.math.multiply_4/Mul/y?
tf.math.multiply_4/MulMul max_pooling2d_4/MaxPool:output:0!tf.math.multiply_4/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_4/Muly
tf.math.multiply_5/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *???2
tf.math.multiply_5/Mul/y?
tf.math.multiply_5/MulMulactivation_5/Relu:activations:0!tf.math.multiply_5/Mul/y:output:0*
T0*0
_output_shapes
:?????????? 2
tf.math.multiply_5/Mulx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2tf.math.multiply_3/Mul:z:0tf.math.multiply_4/Mul:z:0tf.math.multiply_5/Mul:z:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????`2
concatenate_1/concat?
 dense_9/MLCMatMul/ReadVariableOpReadVariableOp)dense_9_mlcmatmul_readvariableop_resource*
_output_shapes

:`0*
dtype02"
 dense_9/MLCMatMul/ReadVariableOp?
dense_9/MLCMatMul	MLCMatMulconcatenate_1/concat:output:0(dense_9/MLCMatMul/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*

input_rank2
dense_9/MLCMatMul?
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02 
dense_9/BiasAdd/ReadVariableOp?
dense_9/BiasAddBiasAdddense_9/MLCMatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
dense_9/BiasAddy
dense_9/TanhTanhdense_9/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
dense_9/Tanh?
max_pooling2d_5/MaxPoolMaxPooldense_9/Tanh:y:0*0
_output_shapes
:??????????0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
tf.compat.v1.squeeze_1/SqueezeSqueeze max_pooling2d_5/MaxPool:output:0*
T0*,
_output_shapes
:??????????0*
squeeze_dims
2 
tf.compat.v1.squeeze_1/Squeezes
lstm_6/ShapeShape'tf.compat.v1.squeeze_1/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_6/Shape?
lstm_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_6/strided_slice/stack?
lstm_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_1?
lstm_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_6/strided_slice/stack_2?
lstm_6/strided_sliceStridedSlicelstm_6/Shape:output:0#lstm_6/strided_slice/stack:output:0%lstm_6/strided_slice/stack_1:output:0%lstm_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_6/strided_slicek
lstm_6/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/mul/y?
lstm_6/zeros/mulMullstm_6/strided_slice:output:0lstm_6/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/mulm
lstm_6/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/Less/y?
lstm_6/zeros/LessLesslstm_6/zeros/mul:z:0lstm_6/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros/Lessq
lstm_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros/packed/1?
lstm_6/zeros/packedPacklstm_6/strided_slice:output:0lstm_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros/packedm
lstm_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros/Const?
lstm_6/zerosFilllstm_6/zeros/packed:output:0lstm_6/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_6/zeroso
lstm_6/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/mul/y?
lstm_6/zeros_1/mulMullstm_6/strided_slice:output:0lstm_6/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/mulq
lstm_6/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/Less/y?
lstm_6/zeros_1/LessLesslstm_6/zeros_1/mul:z:0lstm_6/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_6/zeros_1/Lessu
lstm_6/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/zeros_1/packed/1?
lstm_6/zeros_1/packedPacklstm_6/strided_slice:output:0 lstm_6/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_6/zeros_1/packedq
lstm_6/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_6/zeros_1/Const?
lstm_6/zeros_1Filllstm_6/zeros_1/packed:output:0lstm_6/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_6/zeros_1{
lstm_6/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/MLCLSTM/hidden_size{
lstm_6/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_6/MLCLSTM/output_size?
lstm_6/MLCLSTM/ReadVariableOpReadVariableOp&lstm_6_mlclstm_readvariableop_resource*
_output_shapes
:	0?*
dtype02
lstm_6/MLCLSTM/ReadVariableOp?
lstm_6/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_6_mlclstm_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02!
lstm_6/MLCLSTM/ReadVariableOp_1?
lstm_6/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_6_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_6/MLCLSTM/ReadVariableOp_2?
lstm_6/MLCLSTMMLCLSTM'tf.compat.v1.squeeze_1/Squeeze:output:0#lstm_6/MLCLSTM/hidden_size:output:0#lstm_6/MLCLSTM/output_size:output:0%lstm_6/MLCLSTM/ReadVariableOp:value:0'lstm_6/MLCLSTM/ReadVariableOp_1:value:0'lstm_6/MLCLSTM/ReadVariableOp_2:value:0*
T0*-
_output_shapes
:???????????*
dropout%    *
return_sequences(2
lstm_6/MLCLSTM?
!dense_10/MLCMatMul/ReadVariableOpReadVariableOp*dense_10_mlcmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dense_10/MLCMatMul/ReadVariableOp?
dense_10/MLCMatMul	MLCMatMullstm_6/MLCLSTM:output:0)dense_10/MLCMatMul/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????*

input_rank2
dense_10/MLCMatMul?
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp?
dense_10/BiasAddBiasAdddense_10/MLCMatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????2
dense_10/BiasAddx
dense_10/TanhTanhdense_10/BiasAdd:output:0*
T0*,
_output_shapes
:??????????2
dense_10/Tanh]
lstm_7/ShapeShapedense_10/Tanh:y:0*
T0*
_output_shapes
:2
lstm_7/Shape?
lstm_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_7/strided_slice/stack?
lstm_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_7/strided_slice/stack_1?
lstm_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_7/strided_slice/stack_2?
lstm_7/strided_sliceStridedSlicelstm_7/Shape:output:0#lstm_7/strided_slice/stack:output:0%lstm_7/strided_slice/stack_1:output:0%lstm_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_7/strided_slicej
lstm_7/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros/mul/y?
lstm_7/zeros/mulMullstm_7/strided_slice:output:0lstm_7/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros/mulm
lstm_7/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_7/zeros/Less/y?
lstm_7/zeros/LessLesslstm_7/zeros/mul:z:0lstm_7/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros/Lessp
lstm_7/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros/packed/1?
lstm_7/zeros/packedPacklstm_7/strided_slice:output:0lstm_7/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_7/zeros/packedm
lstm_7/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_7/zeros/Const?
lstm_7/zerosFilllstm_7/zeros/packed:output:0lstm_7/zeros/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/zerosn
lstm_7/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros_1/mul/y?
lstm_7/zeros_1/mulMullstm_7/strided_slice:output:0lstm_7/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros_1/mulq
lstm_7/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_7/zeros_1/Less/y?
lstm_7/zeros_1/LessLesslstm_7/zeros_1/mul:z:0lstm_7/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_7/zeros_1/Lesst
lstm_7/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/zeros_1/packed/1?
lstm_7/zeros_1/packedPacklstm_7/strided_slice:output:0 lstm_7/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_7/zeros_1/packedq
lstm_7/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_7/zeros_1/Const?
lstm_7/zeros_1Filllstm_7/zeros_1/packed:output:0lstm_7/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/zeros_1z
lstm_7/MLCLSTM/hidden_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/MLCLSTM/hidden_sizez
lstm_7/MLCLSTM/output_sizeConst*
_output_shapes
: *
dtype0*
value	B :<2
lstm_7/MLCLSTM/output_size?
lstm_7/MLCLSTM/ReadVariableOpReadVariableOp&lstm_7_mlclstm_readvariableop_resource*
_output_shapes
:	?*
dtype02
lstm_7/MLCLSTM/ReadVariableOp?
lstm_7/MLCLSTM/ReadVariableOp_1ReadVariableOp(lstm_7_mlclstm_readvariableop_1_resource*
_output_shapes
:	<?*
dtype02!
lstm_7/MLCLSTM/ReadVariableOp_1?
lstm_7/MLCLSTM/ReadVariableOp_2ReadVariableOp(lstm_7_mlclstm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02!
lstm_7/MLCLSTM/ReadVariableOp_2?
lstm_7/MLCLSTMMLCLSTMdense_10/Tanh:y:0#lstm_7/MLCLSTM/hidden_size:output:0#lstm_7/MLCLSTM/output_size:output:0%lstm_7/MLCLSTM/ReadVariableOp:value:0'lstm_7/MLCLSTM/ReadVariableOp_1:value:0'lstm_7/MLCLSTM/ReadVariableOp_2:value:0*
T0*+
_output_shapes
:?????????<*
dropout%    2
lstm_7/MLCLSTM}
lstm_7/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????<   2
lstm_7/Reshape/shape?
lstm_7/ReshapeReshapelstm_7/MLCLSTM:output:0lstm_7/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????<2
lstm_7/Reshape?
!dense_11/MLCMatMul/ReadVariableOpReadVariableOp*dense_11_mlcmatmul_readvariableop_resource*
_output_shapes
:	<?*
dtype02#
!dense_11/MLCMatMul/ReadVariableOp?
dense_11/MLCMatMul	MLCMatMullstm_7/Reshape:output:0)dense_11/MLCMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/MLCMatMul?
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_11/BiasAdd/ReadVariableOp?
dense_11/BiasAddBiasAdddense_11/MLCMatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_11/BiasAddt
dense_11/TanhTanhdense_11/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_11/Tanh?
IdentityIdentitydense_11/Tanh:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/MLCMatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/MLCMatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/MLCMatMul/ReadVariableOp^lstm_6/MLCLSTM/ReadVariableOp ^lstm_6/MLCLSTM/ReadVariableOp_1 ^lstm_6/MLCLSTM/ReadVariableOp_2^lstm_7/MLCLSTM/ReadVariableOp ^lstm_7/MLCLSTM/ReadVariableOp_1 ^lstm_7/MLCLSTM/ReadVariableOp_2*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:??????????:??????????:??????????::::::::::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/MLCMatMul/ReadVariableOp!dense_10/MLCMatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/MLCMatMul/ReadVariableOp!dense_11/MLCMatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/MLCMatMul/ReadVariableOp dense_9/MLCMatMul/ReadVariableOp2>
lstm_6/MLCLSTM/ReadVariableOplstm_6/MLCLSTM/ReadVariableOp2B
lstm_6/MLCLSTM/ReadVariableOp_1lstm_6/MLCLSTM/ReadVariableOp_12B
lstm_6/MLCLSTM/ReadVariableOp_2lstm_6/MLCLSTM/ReadVariableOp_22>
lstm_7/MLCLSTM/ReadVariableOplstm_7/MLCLSTM/ReadVariableOp2B
lstm_7/MLCLSTM/ReadVariableOp_1lstm_7/MLCLSTM/ReadVariableOp_12B
lstm_7/MLCLSTM/ReadVariableOp_2lstm_7/MLCLSTM/ReadVariableOp_2:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/2
? 
?
A__inference_lstm_6_layer_call_and_return_conditional_losses_71904

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
:???????????*
dropout%    *
return_sequences(2	
MLCLSTM?
IdentityIdentityMLCLSTM:output:0^MLCLSTM/ReadVariableOp^MLCLSTM/ReadVariableOp_1^MLCLSTM/ReadVariableOp_2*
T0*-
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????0:::20
MLCLSTM/ReadVariableOpMLCLSTM/ReadVariableOp24
MLCLSTM/ReadVariableOp_1MLCLSTM/ReadVariableOp_124
MLCLSTM/ReadVariableOp_2MLCLSTM/ReadVariableOp_2:T P
,
_output_shapes
:??????????0
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71335

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
conv2d_3_input@
 serving_default_conv2d_3_input:0??????????
R
conv2d_4_input@
 serving_default_conv2d_4_input:0??????????
R
conv2d_5_input@
 serving_default_conv2d_5_input:0??????????=
dense_111
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
_tf_keras_network??{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}, "name": "conv2d_3_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}, "name": "conv2d_4_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_3_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_4_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}, "name": "conv2d_5_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_5_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["max_pooling2d_3", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["max_pooling2d_4", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["activation_5", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["tf.math.multiply_3", 0, 0, {}], ["tf.math.multiply_4", 0, 0, {}], ["tf.math.multiply_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze_1", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze_1", "inbound_nodes": [["max_pooling2d_5", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_6", "inbound_nodes": [[["tf.compat.v1.squeeze_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["lstm_6", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_7", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_7", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["lstm_7", 0, 0, {}]]]}], "input_layers": [["conv2d_3_input", 0, 0], ["conv2d_4_input", 0, 0], ["conv2d_5_input", 0, 0]], "output_layers": [["dense_11", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 300, 5]}, {"class_name": "TensorShape", "items": [null, 1, 300, 5]}, {"class_name": "TensorShape", "items": [null, 1, 300, 5]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}, "name": "conv2d_3_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}, "name": "conv2d_4_input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_3_input", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["conv2d_4_input", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}, "name": "conv2d_5_input", "inbound_nodes": []}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_5_input", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_3", "inbound_nodes": [["max_pooling2d_3", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_4", "inbound_nodes": [["max_pooling2d_4", 0, 0, {"y": 0.2, "name": null}]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}, "name": "tf.math.multiply_5", "inbound_nodes": [["activation_5", 0, 0, {"y": 0.6, "name": null}]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["tf.math.multiply_3", 0, 0, {}], ["tf.math.multiply_4", 0, 0, {}], ["tf.math.multiply_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_9", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["dense_9", 0, 0, {}]]]}, {"class_name": "TFOpLambda", "config": {"name": "tf.compat.v1.squeeze_1", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}, "name": "tf.compat.v1.squeeze_1", "inbound_nodes": [["max_pooling2d_5", 0, 0, {"axis": [1]}]]}, {"class_name": "LSTM", "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_6", "inbound_nodes": [[["tf.compat.v1.squeeze_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_10", "inbound_nodes": [[["lstm_6", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_7", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "lstm_7", "inbound_nodes": [[["dense_10", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_11", "inbound_nodes": [[["lstm_7", 0, 0, {}]]]}], "input_layers": [["conv2d_3_input", 0, 0], ["conv2d_4_input", 0, 0], ["conv2d_5_input", 0, 0]], "output_layers": [["dense_11", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_3_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_4_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}
?


kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 300, 5]}}
?


#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 4]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 300, 5]}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "conv2d_5_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
?


1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 300, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 6]}, "strides": {"class_name": "__tuple__", "items": [5, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 300, 5]}}
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
?
C	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_3", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
D	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_4", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
E	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.math.multiply_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.math.multiply_5", "trainable": true, "dtype": "float32", "function": "math.multiply"}}
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 300, 32]}, {"class_name": "TensorShape", "items": [null, 1, 300, 32]}, {"class_name": "TensorShape", "items": [null, 1, 300, 32]}]}
?

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96]}, "dtype": "float32", "units": 48, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 300, 96]}}
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [1, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
T	keras_api"?
_tf_keras_layer?{"class_name": "TFOpLambda", "name": "tf.compat.v1.squeeze_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "tf.compat.v1.squeeze_1", "trainable": true, "dtype": "float32", "function": "compat.v1.squeeze"}}
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
{"class_name": "LSTM", "name": "lstm_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 120, 48]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 48]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 48]}}
?

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_10", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 200]}}
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

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "lstm_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_7", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 30]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 30]}}
?

gkernel
hbias
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_11", "trainable": true, "dtype": "float32", "units": 240, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 60}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 60]}}
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
):' 2conv2d_3/kernel
: 2conv2d_3/bias
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
):' 2conv2d_4/kernel
: 2conv2d_4/bias
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
):' 2conv2d_5/kernel
: 2conv2d_5/bias
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
 :`02dense_9/kernel
:02dense_9/bias
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
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_12", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
": 	?2dense_10/kernel
:2dense_10/bias
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
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_13", "trainable": true, "dtype": "float32", "units": 60, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
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
": 	<?2dense_11/kernel
:?2dense_11/bias
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
-:+	0?2lstm_6/lstm_cell_12/kernel
8:6
??2$lstm_6/lstm_cell_12/recurrent_kernel
':%?2lstm_6/lstm_cell_12/bias
-:+	?2lstm_7/lstm_cell_13/kernel
7:5	<?2$lstm_7/lstm_cell_13/recurrent_kernel
':%?2lstm_7/lstm_cell_13/bias
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
.:, 2Adam/conv2d_3/kernel/m
 : 2Adam/conv2d_3/bias/m
.:, 2Adam/conv2d_4/kernel/m
 : 2Adam/conv2d_4/bias/m
.:, 2Adam/conv2d_5/kernel/m
 : 2Adam/conv2d_5/bias/m
%:#`02Adam/dense_9/kernel/m
:02Adam/dense_9/bias/m
':%	?2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
':%	<?2Adam/dense_11/kernel/m
!:?2Adam/dense_11/bias/m
2:0	0?2!Adam/lstm_6/lstm_cell_12/kernel/m
=:;
??2+Adam/lstm_6/lstm_cell_12/recurrent_kernel/m
,:*?2Adam/lstm_6/lstm_cell_12/bias/m
2:0	?2!Adam/lstm_7/lstm_cell_13/kernel/m
<::	<?2+Adam/lstm_7/lstm_cell_13/recurrent_kernel/m
,:*?2Adam/lstm_7/lstm_cell_13/bias/m
.:, 2Adam/conv2d_3/kernel/v
 : 2Adam/conv2d_3/bias/v
.:, 2Adam/conv2d_4/kernel/v
 : 2Adam/conv2d_4/bias/v
.:, 2Adam/conv2d_5/kernel/v
 : 2Adam/conv2d_5/bias/v
%:#`02Adam/dense_9/kernel/v
:02Adam/dense_9/bias/v
':%	?2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
':%	<?2Adam/dense_11/kernel/v
!:?2Adam/dense_11/bias/v
2:0	0?2!Adam/lstm_6/lstm_cell_12/kernel/v
=:;
??2+Adam/lstm_6/lstm_cell_12/recurrent_kernel/v
,:*?2Adam/lstm_6/lstm_cell_12/bias/v
2:0	?2!Adam/lstm_7/lstm_cell_13/kernel/v
<::	<?2+Adam/lstm_7/lstm_cell_13/recurrent_kernel/v
,:*?2Adam/lstm_7/lstm_cell_13/bias/v
?2?
'__inference_model_1_layer_call_fn_72786
'__inference_model_1_layer_call_fn_72296
'__inference_model_1_layer_call_fn_72403
'__inference_model_1_layer_call_fn_72743?
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
 __inference__wrapped_model_71329?
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
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_72700
B__inference_model_1_layer_call_and_return_conditional_losses_72188
B__inference_model_1_layer_call_and_return_conditional_losses_72578
B__inference_model_1_layer_call_and_return_conditional_losses_72124?
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
(__inference_conv2d_3_layer_call_fn_72805?
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_72796?
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
(__inference_conv2d_4_layer_call_fn_72824?
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_72815?
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
,__inference_activation_3_layer_call_fn_72834?
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
G__inference_activation_3_layer_call_and_return_conditional_losses_72829?
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
,__inference_activation_4_layer_call_fn_72844?
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
G__inference_activation_4_layer_call_and_return_conditional_losses_72839?
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
(__inference_conv2d_5_layer_call_fn_72863?
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_72854?
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
/__inference_max_pooling2d_3_layer_call_fn_71341?
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71335?
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
/__inference_max_pooling2d_4_layer_call_fn_71353?
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_71347?
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
,__inference_activation_5_layer_call_fn_72873?
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
G__inference_activation_5_layer_call_and_return_conditional_losses_72868?
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
-__inference_concatenate_1_layer_call_fn_72888?
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_72881?
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
'__inference_dense_9_layer_call_fn_72908?
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
B__inference_dense_9_layer_call_and_return_conditional_losses_72899?
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
/__inference_max_pooling2d_5_layer_call_fn_71365?
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_71359?
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
&__inference_lstm_6_layer_call_fn_73088
&__inference_lstm_6_layer_call_fn_72998
&__inference_lstm_6_layer_call_fn_72987
&__inference_lstm_6_layer_call_fn_73077?
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
A__inference_lstm_6_layer_call_and_return_conditional_losses_73032
A__inference_lstm_6_layer_call_and_return_conditional_losses_72976
A__inference_lstm_6_layer_call_and_return_conditional_losses_73066
A__inference_lstm_6_layer_call_and_return_conditional_losses_72942?
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
(__inference_dense_10_layer_call_fn_73108?
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
C__inference_dense_10_layer_call_and_return_conditional_losses_73099?
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
&__inference_lstm_7_layer_call_fn_73285
&__inference_lstm_7_layer_call_fn_73296
&__inference_lstm_7_layer_call_fn_73202
&__inference_lstm_7_layer_call_fn_73191?
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_73274
A__inference_lstm_7_layer_call_and_return_conditional_losses_73238
A__inference_lstm_7_layer_call_and_return_conditional_losses_73144
A__inference_lstm_7_layer_call_and_return_conditional_losses_73180?
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
(__inference_dense_11_layer_call_fn_73316?
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
C__inference_dense_11_layer_call_and_return_conditional_losses_73307?
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
#__inference_signature_wrapper_72456conv2d_3_inputconv2d_4_inputconv2d_5_input"?
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
 __inference__wrapped_model_71329?#$12JKrst[\uvwgh???
???
???
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
? "4?1
/
dense_11#? 
dense_11???????????
G__inference_activation_3_layer_call_and_return_conditional_losses_72829j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_3_layer_call_fn_72834]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
G__inference_activation_4_layer_call_and_return_conditional_losses_72839j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_4_layer_call_fn_72844]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
G__inference_activation_5_layer_call_and_return_conditional_losses_72868j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
,__inference_activation_5_layer_call_fn_72873]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_72881????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? ".?+
$?!
0??????????`
? ?
-__inference_concatenate_1_layer_call_fn_72888????
???
???
+?(
inputs/0?????????? 
+?(
inputs/1?????????? 
+?(
inputs/2?????????? 
? "!???????????`?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_72796n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_3_layer_call_fn_72805a8?5
.?+
)?&
inputs??????????
? "!??????????? ?
C__inference_conv2d_4_layer_call_and_return_conditional_losses_72815n#$8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_4_layer_call_fn_72824a#$8?5
.?+
)?&
inputs??????????
? "!??????????? ?
C__inference_conv2d_5_layer_call_and_return_conditional_losses_72854n128?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
(__inference_conv2d_5_layer_call_fn_72863a128?5
.?+
)?&
inputs??????????
? "!??????????? ?
C__inference_dense_10_layer_call_and_return_conditional_losses_73099g[\5?2
+?(
&?#
inputs???????????
? "*?'
 ?
0??????????
? ?
(__inference_dense_10_layer_call_fn_73108Z[\5?2
+?(
&?#
inputs???????????
? "????????????
C__inference_dense_11_layer_call_and_return_conditional_losses_73307]gh/?,
%?"
 ?
inputs?????????<
? "&?#
?
0??????????
? |
(__inference_dense_11_layer_call_fn_73316Pgh/?,
%?"
 ?
inputs?????????<
? "????????????
B__inference_dense_9_layer_call_and_return_conditional_losses_72899nJK8?5
.?+
)?&
inputs??????????`
? ".?+
$?!
0??????????0
? ?
'__inference_dense_9_layer_call_fn_72908aJK8?5
.?+
)?&
inputs??????????`
? "!???????????0?
A__inference_lstm_6_layer_call_and_return_conditional_losses_72942trst@?=
6?3
%?"
inputs??????????0

 
p

 
? "+?(
!?
0???????????
? ?
A__inference_lstm_6_layer_call_and_return_conditional_losses_72976trst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "+?(
!?
0???????????
? ?
A__inference_lstm_6_layer_call_and_return_conditional_losses_73032?rstO?L
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
A__inference_lstm_6_layer_call_and_return_conditional_losses_73066?rstO?L
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
&__inference_lstm_6_layer_call_fn_72987grst@?=
6?3
%?"
inputs??????????0

 
p

 
? "?????????????
&__inference_lstm_6_layer_call_fn_72998grst@?=
6?3
%?"
inputs??????????0

 
p 

 
? "?????????????
&__inference_lstm_6_layer_call_fn_73077~rstO?L
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
&__inference_lstm_6_layer_call_fn_73088~rstO?L
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_73144nuvw@?=
6?3
%?"
inputs??????????

 
p

 
? "%?"
?
0?????????<
? ?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73180nuvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "%?"
?
0?????????<
? ?
A__inference_lstm_7_layer_call_and_return_conditional_losses_73238}uvwO?L
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
A__inference_lstm_7_layer_call_and_return_conditional_losses_73274}uvwO?L
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
&__inference_lstm_7_layer_call_fn_73191auvw@?=
6?3
%?"
inputs??????????

 
p

 
? "??????????<?
&__inference_lstm_7_layer_call_fn_73202auvw@?=
6?3
%?"
inputs??????????

 
p 

 
? "??????????<?
&__inference_lstm_7_layer_call_fn_73285puvwO?L
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
&__inference_lstm_7_layer_call_fn_73296puvwO?L
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_71335?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_3_layer_call_fn_71341?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_71347?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_71353?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_71359?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_5_layer_call_fn_71365?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_72124?#$12JKrst[\uvwgh???
???
???
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_72188?#$12JKrst[\uvwgh???
???
???
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
p 

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_72578?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p

 
? "&?#
?
0??????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_72700?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p 

 
? "&?#
?
0??????????
? ?
'__inference_model_1_layer_call_fn_72296?#$12JKrst[\uvwgh???
???
???
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
p

 
? "????????????
'__inference_model_1_layer_call_fn_72403?#$12JKrst[\uvwgh???
???
???
1?.
conv2d_3_input??????????
1?.
conv2d_4_input??????????
1?.
conv2d_5_input??????????
p 

 
? "????????????
'__inference_model_1_layer_call_fn_72743?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p

 
? "????????????
'__inference_model_1_layer_call_fn_72786?#$12JKrst[\uvwgh???
???
???
+?(
inputs/0??????????
+?(
inputs/1??????????
+?(
inputs/2??????????
p 

 
? "????????????
#__inference_signature_wrapper_72456?#$12JKrst[\uvwgh???
? 
???
C
conv2d_3_input1?.
conv2d_3_input??????????
C
conv2d_4_input1?.
conv2d_4_input??????????
C
conv2d_5_input1?.
conv2d_5_input??????????"4?1
/
dense_11#? 
dense_11??????????