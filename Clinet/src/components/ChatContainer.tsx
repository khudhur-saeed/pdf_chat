import { useState, useEffect, useRef } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { TypingIndicator } from "./TypingIndicator";
import { ScrollArea } from "./ui/scroll-area";

const API_BASE_URL = "http://localhost:8000";

export interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  attachedFile?: File;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  sessionId?: string; // Backend session ID
}

interface ChatContainerProps {
  currentChat: Chat | null;
  onUpdateChat: (chat: Chat) => void;
}

// API helper functions
async function createSession(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/create-session`, {
    method: "POST",
  });
  const data = await response.json();
  return data.session_id;
}

async function sendMessage(sessionId: string, message: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message,
    }),
  });
  const data = await response.json();
  return data.response;
}

async function uploadPdf(file: File): Promise<boolean> {
  const formData = new FormData();
  formData.append("file", file);
  
  const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
    method: "POST",
    body: formData,
  });
  return response.ok;
}

export function ChatContainer({ currentChat, onUpdateChat }: ChatContainerProps) {
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentChat?.messages, isTyping]);

  const handleSendMessage = async (messageText: string, attachedFile?: File) => {
    if (!currentChat) return;

    // Ensure we have a session ID
    let sessionId = currentChat.sessionId;
    if (!sessionId) {
      try {
        sessionId = await createSession();
      } catch (error) {
        console.error("Failed to create session:", error);
        return;
      }
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      isUser: true,
      timestamp: new Date(),
      attachedFile: attachedFile,
    };
    
    const updatedMessages = [...currentChat.messages, userMessage];
    
    // Update chat title if this is the first user message after initial
    let updatedTitle = currentChat.title;
    if (currentChat.messages.length <= 1) {
      updatedTitle = messageText.slice(0, 50) + (messageText.length > 50 ? "..." : "");
    }
    
    const updatedChat: Chat = {
      ...currentChat,
      title: updatedTitle,
      messages: updatedMessages,
      updatedAt: new Date(),
      sessionId: sessionId,
    };
    
    onUpdateChat(updatedChat);
    setIsTyping(true);

    try {
      // Call the backend API
      const aiResponse = await sendMessage(sessionId, messageText);
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        isUser: false,
        timestamp: new Date(),
      };
      
      const finalChat: Chat = {
        ...updatedChat,
        messages: [...updatedMessages, aiMessage],
        updatedAt: new Date(),
      };
      
      onUpdateChat(finalChat);
    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, there was an error connecting to the server. Please try again.",
        isUser: false,
        timestamp: new Date(),
      };
      
      const finalChat: Chat = {
        ...updatedChat,
        messages: [...updatedMessages, errorMessage],
        updatedAt: new Date(),
      };
      
      onUpdateChat(finalChat);
    } finally {
      setIsTyping(false);
    }
  };

  if (!currentChat) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <div className="max-w-md">
          <h2 className="text-xl mb-4 text-foreground">Welcome to PDF Chat</h2>
          <p className="text-muted-foreground mb-6">
            Upload a PDF and start a new conversation to begin asking questions about your document.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">
      {/* Messages */}
      <ScrollArea ref={scrollAreaRef} className="flex-1 h-0">
        <div className="max-w-3xl mx-auto w-full pb-4">
          {currentChat.messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message.text}
              isUser={message.isUser}
              timestamp={message.timestamp}
            />
          ))}
          {isTyping && <TypingIndicator />}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="flex-shrink-0 p-3 sm:p-4 md:p-6 border-t border-border">
        <div className="max-w-3xl mx-auto w-full">
          <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
        </div>
      </div>
    </div>
  );
}