import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {AuthComponent} from "./Component/auth/auth.component";

const routes: Routes = [

    {
        path: '',
        component: AuthComponent,
        children: [
            {
                path: 'display-auth',
                loadChildren: () => import('./Modules/DisplayAuth/display-auth.module').then(m => m.DisplayAuthModule),
            },
            {
                path: 'sign-in',
                loadChildren: () => import('./Modules/SignIn/sign-in.module').then(m => m.SignInModule),
            },
            {
                path: 'sign-up',
                loadChildren: () => import('./Modules/SignUp/sign-up.module').then(m => m.SignUpModule),
            }
        ]
    },
];

@NgModule({
    imports: [RouterModule.forChild(routes)],
    exports: [RouterModule]
})
export class AuthRoutingModule {
}
