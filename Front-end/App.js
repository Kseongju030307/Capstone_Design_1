import React from 'react';
import { createStackNavigator, TransitionPresets } from '@react-navigation/stack';
import { NavigationContainer } from '@react-navigation/native';

// 화면 import
import HomeScreen from './screens/HomeScreen';
import UploadScreen from './screens/UploadScreen';
import ResultsScreen from './screens/ResultsScreen';

const Stack = createStackNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#1E1E1E',  // 헤더 배경색
          },
          headerTintColor: '#FFFFFF',   // 헤더 글자색
          headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 20,
          },
          ...TransitionPresets.SlideFromRightIOS,  // 화면 전환 애니메이션
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} options={{ title: '🏠 홈 화면' }} />
        <Stack.Screen name="Upload" component={UploadScreen} options={{ title: '📤 스케치 업로드' }} />
        <Stack.Screen name="Results" component={ResultsScreen} options={{ title: '📊 결과 화면' }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
