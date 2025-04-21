import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';

const routes: Routes = [
    {
        path: '',
        redirectTo: '/display-auth',
        pathMatch: 'full',
    },
    {
        path: '',
        loadChildren: () => import('./Modules/Auth/auth.module').then(m => m.AuthModule)
    },
    {
        path: 'main',
        loadChildren: () => import('./Modules/Main/main.module').then(m => m.MainModule)
    }
];

@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
})
export class AppRoutingModule {
}
