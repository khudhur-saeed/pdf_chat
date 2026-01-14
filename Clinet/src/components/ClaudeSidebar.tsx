import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Sidebar, SidebarContent, SidebarHeader, SidebarFooter } from "./ui/sidebar";
import { UploadedFiles } from "./UploadedFiles";
import { 
  Plus, 
  ChevronDown,
  User,
  Trash2,
  History
} from "lucide-react";

export interface ChatHistory {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messageCount: number;
}

interface ClaudeSidebarProps {
  chatHistory: ChatHistory[];
  currentChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onDeleteChat: (chatId: string) => void;
  onClearAllChats?: () => void;
  uploadRefreshTrigger?: number;
  selectedPdf?: string;
  onPdfSelected?: (filename: string) => void;
  onPdfDeleted?: () => void;
}

export function ClaudeSidebar({ 
  chatHistory, 
  currentChatId, 
  onSelectChat, 
  onNewChat, 
  onDeleteChat,
  onClearAllChats,
  uploadRefreshTrigger,
  selectedPdf,
  onPdfSelected,
  onPdfDeleted
}: ClaudeSidebarProps) {
  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return "Today";
    if (days === 1) return "Yesterday";
    if (days < 7) return `${days}d`;
    return date.toLocaleDateString();
  };

  return (
    <Sidebar className="w-64 border-r border-sidebar-border hidden lg:flex">
      <SidebarHeader className="p-3 sm:p-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2 mb-3 sm:mb-4">
          <div className="w-6 h-6 bg-orange-500 rounded flex items-center justify-center">
            <span className="text-white text-xs font-medium">C</span>
          </div>
          <h1 className="text-lg font-medium text-sidebar-foreground">CHAT-PDF</h1>
        </div>
        
        <div className="flex gap-2">
          <Button 
            onClick={onNewChat} 
            className="flex-1 justify-start bg-sidebar-accent hover:bg-sidebar-accent/80 text-sidebar-foreground border border-sidebar-border text-xs sm:text-sm" 
            size="sm"
          >
            <Plus className="h-4 w-4 mr-2" />
            New chat
          </Button>
          
          {chatHistory.length > 0 && onClearAllChats && (
            <Button
              onClick={onClearAllChats}
              variant="ghost"
              size="sm"
              className="px-2 sm:px-3 hover:bg-sidebar-accent text-sidebar-foreground/70 hover:text-sidebar-foreground"
              title="Clear all chats"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </SidebarHeader>

      <SidebarContent className="p-0">
        <ScrollArea className="flex-1">
          <div className="p-2">
            {/* Uploaded PDFs Section */}
            <div className="mb-4">
              <UploadedFiles 
                refreshTrigger={uploadRefreshTrigger} 
                selectedFile={selectedPdf}
                onFileSelected={onPdfSelected}
                onFileDeleted={onPdfDeleted}
              />
            </div>

            {/* Chat History Section */}
            {chatHistory.length > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 px-2 py-1 text-xs font-medium text-sidebar-foreground/70 uppercase tracking-wider">
                  <History className="h-3 w-3" />
                  Chat History
                </div>
                <div className="space-y-1 mt-2">
                  {chatHistory.map((chat) => (
                    <div
                      key={chat.id}
                      className={`group relative w-full text-left px-2 py-1.5 text-sm rounded-md cursor-pointer ${
                        currentChatId === chat.id
                          ? "bg-sidebar-accent text-sidebar-foreground"
                          : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50"
                      }`}
                      onClick={() => onSelectChat(chat.id)}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate flex-1">{chat.title}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteChat(chat.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 hover:text-red-500 transition-opacity"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                      <div className="text-xs text-sidebar-foreground/50 mt-0.5">
                        {formatTime(chat.timestamp)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-sidebar-border">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-orange-500 rounded-full flex items-center justify-center">
            <User className="h-4 w-4 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-sidebar-foreground">User</p>
            <p className="text-xs text-sidebar-foreground/70">Active</p>
          </div>
          <ChevronDown className="h-4 w-4 text-sidebar-foreground/70" />
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}