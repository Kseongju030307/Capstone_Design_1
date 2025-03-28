import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const UploadScreen = ({ navigation }) => {
  const [imageUri, setImageUri] = useState(null);

  const pickImage = async () => {
    // 📌 갤러리 접근 권한 요청
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('갤러리 접근 권한이 필요합니다.');
      return;
    }

    // 📌 갤러리에서 이미지 선택
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,  // 선택한 이미지 편집 가능
      quality: 1,  // 이미지 품질 설정 (1 = 최고 화질)
    });
    
    if (!result.canceled) {
      const selectedImage = result.assets[0].uri;
      setImageUri(selectedImage);  // 선택한 이미지 저장

      // 📌 이미지 선택 후 ResultsScreen으로 이동
      navigation.navigate('Results', { imageUri: selectedImage });
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>스케치 업로드</Text>

      <TouchableOpacity style={styles.button} onPress={pickImage}>
        <Text style={styles.buttonText}>이미지 선택</Text>
      </TouchableOpacity>

      {imageUri && <Image source={{ uri: imageUri }} style={styles.image} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 10,
    marginBottom: 20,
  },
  buttonText: {
    fontSize: 18,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 10,
    marginTop: 10,
  },
});

export default UploadScreen;
