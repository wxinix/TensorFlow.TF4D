// MIT License
// Copyright (c) 2021 Wuping Xin.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

unit CApiTest;

interface

uses
  DUnitX.TestFramework;

type
  [TestFixture]
  TTensorFlowApiTestObject = class
  private
    class procedure BufferDeallocator(data: Pointer; length: NativeInt); cdecl; static;
    class procedure TensorDeallocator(data: Pointer; len: NativeInt; arg: Pointer); cdecl; static;
  public
    [Setup]
    procedure Setup;
    [TearDown]
    procedure TearDown;
    [Test]
    procedure Test_Version;
    [Test]
    procedure Test_NewBufferFromString;
    [Test]
    procedure Test_NewBuffer;
    [Test]
    procedure Test_GetBuffer;
    [Test]
    procedure Test_Allocate_Tensor;
    [Test]
    procedure Test_New_Tensor;
  end;

implementation

uses
  TF4D.Core.CApi;

class procedure TTensorFlowApiTestObject.BufferDeallocator(data: Pointer; length: NativeInt);
begin
  System.Writeln(#9'Deleting data, length: ', length);
end;

procedure TTensorFlowApiTestObject.Setup;
begin
end;

procedure TTensorFlowApiTestObject.TearDown;
begin
end;

class procedure TTensorFlowApiTestObject.TensorDeallocator(data: Pointer; len: NativeInt; arg: Pointer);
begin
  FreeMem(data);
  System.Writeln(#9'Releasing tensor bytes ', len);
end;

procedure TTensorFlowApiTestObject.Test_Allocate_Tensor;
begin
  var dims: TArray<Int64> := [1, 5, 12];
  var data_size: NativeInt := SizeOf(Single);
  for var I := Low(dims) to High(dims) do
    data_size := data_size * dims[I];

  var data: TArray<Single> :=
    [
      -0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428,
      -0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465,
      -0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230,
      -0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635,
      -0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039
    ];

  var tensor := TF_AllocateTensor(TF_DataType.TF_FLOAT, PInt64(dims), Length(dims), data_size);
  Assert.IsTrue(Assigned(tensor));
  Assert.AreEqual(TF_TensorByteSize(tensor), data_size);
  Move(data[0], TF_TensorData(tensor)^, data_size);
  Assert.AreEqual(TF_DataType.TF_FLOAT, TF_TensorType(tensor));
  Assert.AreEqual(TF_NumDims(tensor), 3);
  Assert.AreEqual(TF_TensorByteSize(tensor), data_size);

  var tensor_data := PSingle(TF_TensorData(tensor));
  {$POINTERMATH ON}
  Assert.AreEqual(tensor_data[1], data[1]);
  {$POINTERMATH OFF}
  TF_DeleteTensor(tensor);
end;

procedure TTensorFlowApiTestObject.Test_GetBuffer;
const
  sTestStr: AnsiString = '123456789012';
  cExpectedLen: NativeInt = 12;
begin
  var LBuffer := TF_NewBuffer;
  LBuffer.data := PAnsiChar(sTestStr);
  LBuffer.Length := Length(sTestStr);
  LBuffer.data_deallocator := BufferDeallocator;
  var LBufferRecord := TF_GetBuffer(LBuffer);
  Assert.AreEqual(LBuffer.data, LBufferRecord.data);
  TF_DeleteBuffer(LBuffer);
end;

procedure TTensorFlowApiTestObject.Test_NewBuffer;
const
  sTestStr: AnsiString = '123456789012';
  cExpectedLen: NativeInt = 12;
begin
  var LBuffer := TF_NewBuffer;
  LBuffer.data := PAnsiChar(sTestStr);
  LBuffer.Length := Length(sTestStr);
  LBuffer.data_deallocator := BufferDeallocator;
  var LBufferRecord := TF_GetBuffer(LBuffer);
  TF_DeleteBuffer(LBuffer);
end;

procedure TTensorFlowApiTestObject.Test_NewBufferFromString;
const
  sTestStr: AnsiString = '123456789012';
  cExpectedLen: NativeInt = 12;
begin
  var LBuffer := TF_NewBufferFromString(PAnsiChar(sTestStr), Length(sTestStr));
  Assert.AreEqual(cExpectedLen, LBuffer.length);

  var LAnsiStr: AnsiString;
  SetLength(LAnsiStr, cExpectedLen);
  Move(PAnsiChar(LBuffer.data)^, LAnsiStr[1], cExpectedLen);
  Assert.AreEqual(sTestStr, LAnsiStr);

  TF_DeleteBuffer(LBuffer);
end;

procedure TTensorFlowApiTestObject.Test_New_Tensor;
begin
  var dims: TArray<Int64> := [1, 5, 12];
  var data_size: NativeInt := SizeOf(Single);
  for var I := Low(dims) to High(dims) do
    data_size := data_size * dims[I];

  var data: PSingle;
  GetMem(data, data_size);

  var values: TArray<Single> :=
    [
      -0.4809832, -0.3770838, 0.1743573, 0.7720509, -0.4064746, 0.0116595, 0.0051413, 0.9135732, 0.7197526, -0.0400658, 0.1180671, -0.6829428,
      -0.4810135, -0.3772099, 0.1745346, 0.7719303, -0.4066443, 0.0114614, 0.0051195, 0.9135003, 0.7196983, -0.0400035, 0.1178188, -0.6830465,
      -0.4809143, -0.3773398, 0.1746384, 0.7719052, -0.4067171, 0.0111654, 0.0054433, 0.9134697, 0.7192584, -0.0399981, 0.1177435, -0.6835230,
      -0.4808300, -0.3774327, 0.1748246, 0.7718700, -0.4070232, 0.0109549, 0.0059128, 0.9133330, 0.7188759, -0.0398740, 0.1181437, -0.6838635,
      -0.4807833, -0.3775733, 0.1748378, 0.7718275, -0.4073670, 0.0107582, 0.0062978, 0.9131795, 0.7187147, -0.0394935, 0.1184392, -0.6840039
    ];

  Move(values[0], data^, data_size);

  var tensor := TF_NewTensor(TF_DataType.TF_FLOAT, PInt64(dims), Length(dims), data, data_size, TensorDeallocator, nil);
  Assert.IsTrue(Assigned(tensor));
  Assert.AreEqual(TF_TensorByteSize(tensor), data_size);
  Assert.AreEqual(TF_DataType.TF_FLOAT, TF_TensorType(tensor));
  Assert.AreEqual(TF_NumDims(tensor), 3);
  Assert.AreEqual(TF_TensorByteSize(tensor), data_size);

  var tensor_data := PSingle(TF_TensorData(tensor));
  {$POINTERMATH ON}
  Assert.AreEqual(tensor_data[1], data[1]);
  {$POINTERMATH OFF}
  TF_DeleteTensor(tensor);
end;

procedure TTensorFlowApiTestObject.Test_Version;
begin
  var LVersion := String(TF_Version());
  Assert.AreEqual(TensorFlowVer, LVersion);
end;

initialization
  TDUnitX.RegisterTestFixture(TTensorFlowApiTestObject);

end.
