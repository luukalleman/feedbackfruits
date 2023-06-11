import React, { useState } from 'react';
import { Button, Text, View } from 'react-native';
import DocumentPicker from 'react-native-document-picker';
import axios from 'axios';

const App = () => {
  const [question, setQuestion] = useState('');
  const [knowledgeBase, setKnowledgeBase] = useState(null);
  const [response, setResponse] = useState(null);

  const uploadPdf = async () => {
    try {
      const res = await DocumentPicker.pick({
        type: [DocumentPicker.types.pdf],
      });

      const data = new FormData();
      data.append('file', {
        uri: res.uri,
        type: res.type,
        name: res.name,
      });

      const config = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };

      axios.post('http://localhost:8000/upload_pdf', data, config)
        .then(response => {
          setKnowledgeBase(response.data.knowledge_base);
        })
        .catch(error => {
          console.error("There was an error uploading the file: ", error);
        });
    } catch (err) {
      console.error('Failed to pick a document: ', err);
    }
  };

  const askQuestion = () => {
    axios.post('http://localhost:8000/ask_question', {
      knowledge_base: knowledgeBase,
      question: question,
    })
    .then(response => {
      setResponse(response.data.response);
    })
    .catch(error => {
      console.error("There was an error asking the question: ", error);
    });
  };

  return (
    <View>
      <Button onPress={uploadPdf} title="Upload a PDF" />
      <TextInput
        onChangeText={text => setQuestion(text)}
        placeholder="Ask a question"
      />
      <Button onPress={askQuestion} title="Ask" />
      {response && <Text>{response}</Text>}
    </View>
  );
};

export default App;
