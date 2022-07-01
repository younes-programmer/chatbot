# IA Chatbot
## About us
> We are a group of developers in ICT department in CRMEF-AGADIR, we are interested in **education and technology** field 
## UML
> ### Sequence diagram
```mermaid
sequenceDiagram
    activate user
    user->>chatbotUI: Hello!
    activate chatbotUI
    deactivate chatbotUI
    activate chatbotUI
    chatbotUI->>MachineLearning: Compare user input with chatbot database!
    activate MachineLearning
    deactivate chatbotUI
    MachineLearning->>ChatbotDatabase: Search for suitable reply
    activate ChatbotDatabase
    deactivate ChatbotDatabase
    MachineLearning-->>chatbotUI: gives back the most suitable reply
    activate chatbotUI
    deactivate MachineLearning
    chatbotUI-->>user: return Input
    deactivate chatbotUI
    deactivate user 
```