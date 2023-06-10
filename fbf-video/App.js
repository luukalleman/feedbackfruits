import React, { useState } from 'react';
import { Button, TextInput, Text, StyleSheet, View } from 'react-native';
import RNFS from 'react-native-fs';
import axios from 'axios';
import fetch from 'node-fetch';

async function fetchPdfFromInternet(url, path) {
    const { data } = await axios.get(url, { responseType: 'arraybuffer' });
    await RNFS.writeFile(path, new Buffer(data, 'binary'), 'binary');
    return path;
}

async function query(data) {
    try {
        const response = await fetch(
            "http://localhost:3000/api/v1/prediction/7df3e69c-9d10-4e58-9302-fd64ed28eb96",
            {
                method: "POST",
                headers: { 
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            }
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        } else {
            const result = await response.json();
            return result;
        }
    } catch (error) {
        console.error("There was an error!", error);
        return { error: error.message }; // Return an object with error message
    }
}

export default function App() {
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");
    const [documentUri, setDocumentUri] = useState("");

    const handlePress = async () => {
        const documentUrl = "https://www.orimi.com/pdf-test.pdf"; // replace with actual URL
        const localPath = `${RNFS.DocumentDirectoryPath}/document.pdf`;
        const localUri = await fetchPdfFromInternet(documentUrl, localPath);

        setDocumentUri(localUri);

        const response = await query({ question: question, document: localUri });
        if (response.error) {
            alert(response.error);
        } else {
            setAnswer(response.answer);
        }
    };

    return (
        <View style={styles.container}>
            <Text style={styles.text}>Document: {documentUri}</Text>
            <TextInput
                style={styles.input}
                placeholder="Enter your question"
                onChangeText={text => setQuestion(text)}
                defaultValue={question}
            />
            <Button title="Get answer" onPress={handlePress} />
            {answer && <Text style={styles.text}>Answer: {answer}</Text>}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        padding: 20,
        backgroundColor: '#f5f5f5',
    },
    input: {
        height: 40,
        borderColor: 'gray',
        borderWidth: 1,
        borderRadius: 5,
        padding: 10,
        marginBottom: 20,
    },
    text: {
        marginVertical: 10,
    },
});
