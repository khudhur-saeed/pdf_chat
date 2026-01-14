import { useState } from "react";
import { ClaudeSidebar, ChatHistory } from "./ClaudeSidebar";
import { ChatContainer, Chat as ChatType, Message } from "./ChatContainer";
import { ClaudeWelcome } from "./ClaudeWelcome";
import { SidebarProvider, SidebarTrigger } from "./ui/sidebar";
import { PdfUpload } from "./PdfUpload";
import { Menu } from "lucide-react";
import { StatusAlert, AlertType } from "./StatusAlert";

const API_BASE_URL = "http://localhost:8000";

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

export function Chat() {
  const [chats, setChats] = useState<ChatType[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [statusAlert, setStatusAlert] = useState<{ type: AlertType; message: string } | null>(null);
  const [uploadRefreshTrigger, setUploadRefreshTrigger] = useState(0);
  const [selectedPdf, setSelectedPdf] = useState<string | null>(null);

  const currentChat = chats.find(chat => chat.id === currentChatId) || null;

  const generateChatHistory = (): ChatHistory[] => {
    return chats.map(chat => ({
      id: chat.id,
      title: chat.title,
      lastMessage: chat.messages.length > 0 
        ? chat.messages[chat.messages.length - 1].text 
        : "New chat",
      timestamp: chat.updatedAt,
      messageCount: chat.messages.filter(m => m.isUser).length,
    }));
  };

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Date.now().toString(),
      title: "New Chat",
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
  };

  const handleSelectChat = (chatId: string) => {
    setCurrentChatId(chatId);
  };

  const handleUpdateChat = (updatedChat: Chat) => {
    setChats(prev => prev.map(chat => 
      chat.id === updatedChat.id ? updatedChat : chat
    ));
  };

  const handleDeleteChat = (chatId: string) => {
    setChats(prev => prev.filter(chat => chat.id !== chatId));
    if (currentChatId === chatId) {
      const remainingChats = chats.filter(chat => chat.id !== chatId);
      setCurrentChatId(remainingChats.length > 0 ? remainingChats[0].id : null);
    }
  };

  const handleClearAllChats = () => {
    if (window.confirm('Are you sure you want to clear all chat history?')) {
      setChats([]);
      setCurrentChatId(null);
      setStatusAlert({ type: "success", message: "All chats cleared successfully" });
      setTimeout(() => setStatusAlert(null), 3000);
    }
  };

  const handlePdfUploadSuccess = (filename: string) => {
    setStatusAlert({ type: "success", message: `PDF "${filename}" uploaded successfully!` });
    setUploadRefreshTrigger(prev => prev + 1); // Trigger refresh of uploaded files list
    setSelectedPdf(filename); // Auto-select the newly uploaded PDF
    setTimeout(() => setStatusAlert(null), 5000);
  };

  const handlePdfSelected = (filename: string) => {
    setSelectedPdf(filename);
    setStatusAlert({ type: "success", message: `Now chatting about: ${filename}` });
    setTimeout(() => setStatusAlert(null), 3000);
  };

  const handlePdfDeleted = () => {
    if (selectedPdf) {
      setSelectedPdf(null);
    }
  };

  const handleWelcomeMessage = async (messageText: string) => {
    // Create a new session
    let sessionId: string;
    try {
      sessionId = await createSession();
    } catch (error) {
      setStatusAlert({ type: "error", message: "Failed to connect to server" });
      setTimeout(() => setStatusAlert(null), 3000);
      return;
    }

    // Create a new chat and send the message
    const newChat: ChatType = {
      id: Date.now().toString(),
      title: messageText.slice(0, 50) + (messageText.length > 50 ? "..." : ""),
      messages: [
        {
          id: Date.now().toString(),
          text: messageText,
          isUser: true,
          timestamp: new Date(),
        }
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
      sessionId: sessionId,
    };

    setChats(prev => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    
    // Get AI response from backend
    try {
      const aiResponse = await sendMessage(sessionId, messageText);
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        isUser: false,
        timestamp: new Date(),
      };
      
      const updatedChat: ChatType = {
        ...newChat,
        messages: [...newChat.messages, aiMessage],
        updatedAt: new Date(),
      };
      
      setChats(prev => prev.map(chat => 
        chat.id === newChat.id ? updatedChat : chat
      ));
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, there was an error connecting to the server. Please try again.",
        isUser: false,
        timestamp: new Date(),
      };
      
      const updatedChat: ChatType = {
        ...newChat,
        messages: [...newChat.messages, errorMessage],
        updatedAt: new Date(),
      };
      
      setChats(prev => prev.map(chat => 
        chat.id === newChat.id ? updatedChat : chat
      ));
    }
  };

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full bg-background overflow-hidden">
        <ClaudeSidebar
          chatHistory={generateChatHistory()}
          currentChatId={currentChatId}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          onDeleteChat={handleDeleteChat}
          onClearAllChats={handleClearAllChats}
          uploadRefreshTrigger={uploadRefreshTrigger}
          selectedPdf={selectedPdf}
          onPdfSelected={handlePdfSelected}
          onPdfDeleted={handlePdfDeleted}
        />
        
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Header with sidebar toggle */}
          <div className="flex-shrink-0 flex items-center gap-2 px-3 sm:px-4 py-3 border-b border-border bg-background">
            <SidebarTrigger className="h-8 w-8 p-0 hover:bg-accent text-[rgba(255,255,255,1)] lg:hidden">
              <Menu className="h-4 w-4" />
            </SidebarTrigger>
            
            <div className="flex-1 min-w-0">
              {currentChat && (
                <h1 className="text-sm font-medium text-foreground truncate">{currentChat.title}</h1>
              )}
              {selectedPdf && (
                <p className="text-xs text-muted-foreground truncate">📄 {selectedPdf}</p>
              )}
            </div>
            
            {/* PDF Upload Button */}
            <div className="ml-auto">
              <PdfUpload 
                onUploadSuccess={handlePdfUploadSuccess}
                onUploadError={(error) => {
                  setStatusAlert({ type: "error", message: error });
                  setTimeout(() => setStatusAlert(null), 5000);
                }}
              />
            </div>
          </div>

          {/* Status Alert */}
          {statusAlert && (
            <div className="flex-shrink-0 px-3 sm:px-4 pt-3">
              <StatusAlert
                type={statusAlert.type}
                message={statusAlert.message}
                onClose={() => setStatusAlert(null)}
              />
            </div>
          )}
          
          <div className="flex-1 overflow-hidden">
            {currentChat && currentChat.messages.length > 0 ? (
              <ChatContainer
                currentChat={currentChat}
                onUpdateChat={handleUpdateChat}
              />
            ) : (
              <ClaudeWelcome onSendMessage={handleWelcomeMessage} />
            )}
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}