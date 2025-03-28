import React, { useRef } from 'react';
import { View, Text, TouchableOpacity, Image, StyleSheet, FlatList, Dimensions } from 'react-native';
import ViewShot from "react-native-view-shot";
import * as Sharing from "expo-sharing";
import * as FileSystem from "expo-file-system";
import { ScrollView } from 'react-native-gesture-handler';

const { height, width } = Dimensions.get("window");

// 탐지된 객체 리스트
const detectedObjects = [
    { id: 1, name: '고양이', confidence: 98, emoji: '🐱' },
    { id: 2, name: '자동차', confidence: 92, emoji: '🚗' },
    { id: 3, name: '의자', confidence: 87, emoji: '🪑' }
];

const ResultsScreen = ({ route, navigation }) => {
    const { imageUri } = route.params;
    const viewShotRef = useRef();

    // 캡처 후 공유 기능
    const captureAndShare = async () => {
        try {
            const uri = await viewShotRef.current.capture();
            const fileUri = FileSystem.cacheDirectory + "result.png";
            await FileSystem.copyAsync({ from: uri, to: fileUri });

            if (await Sharing.isAvailableAsync()) {
                await Sharing.shareAsync(fileUri);
            } else {
                alert("이 기기에서는 공유 기능을 지원하지 않습니다.");
            }
        } catch (error) {
            console.error("공유 실패:", error);
        }
    };

    return (
        <View style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContainer}>
                {/* 캡처할 영역 */}
                <ViewShot ref={viewShotRef} options={{ format: "png", quality: 0.9 }}>
                    <Text style={styles.title}>분석 결과</Text>
                    <Image source={{ uri: imageUri }} style={styles.image} />
                    <Text style={styles.subtitle}>탐지된 객체</Text>

                    {/* `FlatList`를 `View` 내부에 두어 ScrollView와 충돌하지 않게 변경 */}
                    <View style={styles.flatListContainer}>
                        <FlatList
                            data={detectedObjects}
                            keyExtractor={(item) => item.id.toString()}
                            renderItem={({ item }) => (
                                <View style={styles.resultCard}>
                                    <Text style={styles.emoji}>{item.emoji}</Text>
                                    <View style={styles.resultText}>
                                        <Text style={styles.objectName}>{item.name}</Text>
                                        <Text style={styles.confidence}>신뢰도: {item.confidence}%</Text>
                                    </View>
                                </View>
                            )}
                            scrollEnabled={false} // 내부 스크롤 비활성화 (ScrollView와 충돌 방지)
                        />
                    </View>
                </ViewShot>
            </ScrollView>

            {/* 버튼 위치 조정 (반응형 UI 적용) */}
            <View style={styles.buttonContainer}>
                <TouchableOpacity style={[styles.button, styles.uploadButton]} onPress={() => navigation.navigate('Upload')}>
                    <Text style={styles.buttonText}>다시 업로드</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.button, styles.shareButton]} onPress={captureAndShare}>
                    <Text style={styles.buttonText}>결과 공유</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f8f9fa',
        padding: 20,
    },
    scrollContainer: {
        alignItems: 'center',
        paddingBottom: 100, // 여백 추가
    },
    title: {
        fontSize: 28,
        fontWeight: 'bold',
        color: '#343a40',
        textAlign: 'center',
        marginBottom: 20,
    },
    image: {
        width: width * 0.9, // 화면 크기에 맞게 조정
        height: height * 0.4,
        borderRadius: 10,
        marginBottom: 20,
        borderWidth: 2,
        borderColor: '#ccc',
    },
    subtitle: {
        fontSize: 22,
        fontWeight: '600',
        color: '#495057',
        marginBottom: 10,
        textAlign: 'center',
    },
    flatListContainer: {
        width: width * 0.9,
        paddingBottom: 10,
    },
    resultCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#ffffff',
        padding: 15,
        marginVertical: 8,
        borderRadius: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
        elevation: 3,
    },
    emoji: {
        fontSize: 30,
        marginRight: 10,
    },
    resultText: {
        marginLeft: 15,
    },
    objectName: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#343a40',
    },
    confidence: {
        fontSize: 18,
        color: '#007AFF',
    },
    buttonContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        marginBottom: 20,
    },
    button: {
        paddingVertical: 14,
        paddingHorizontal: 24,
        borderRadius: 10,
        marginHorizontal: 8,
    },
    uploadButton: {
        backgroundColor: '#007AFF',
    },
    shareButton: {
        backgroundColor: '#34C759',
    },
    buttonText: {
        fontSize: 18,
        color: '#ffffff',
        fontWeight: '600',
    },
});

export default ResultsScreen;